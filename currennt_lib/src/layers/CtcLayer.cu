#include "CtcLayer.hpp"

namespace internal
{
namespace
{
    class Log
    {
        real_t expVal,logVal;
    public:
        static real_t expMax,expMin,expLimit,logZero,logInfinity;
        static real_t safe_exp(real_t x)
        {
            if (x==logZero)
                return(0);
            if (x>=expLimit)
                return(expMax);
            return(std::exp(x));
        }
        static real_t safe_log(real_t x)
        {
            if (x<expMin)
                return(logZero);
            return(std::log(x));
        }
        static real_t log_add(real_t x,real_t y)
        {
            if (x==logZero)
                return(y);
            if (y==logZero)
                return(x);
            if (x<y)
                thrust::swap(x,y);
            return(x+std::log(1.0+safe_exp(y-x)));
        }
        static real_t log_subtract(real_t x,real_t y)
        {
            if (y==logZero)
                return(x);
            if (y>=x)
                return(logZero);
            return(x+std::log(1.0-safe_exp(y-x)));
        }
        static real_t log_multiply(real_t x,real_t y)
        {
            if (x==logZero || y==logZero)
                return(logZero);
            return(x+y);
        }
        static real_t log_divide(real_t x,real_t y)
        {
            if (x==logZero)
                return(logZero);
            if (y==logZero)
                return(logInfinity);
            return(x-y);
        }
        Log(real_t v=0,bool logScale=false):expVal(logScale?-1:v),logVal(logScale?v:safe_log(v)){}
        Log &operator =(const Log &l)
        {
            logVal=l.logVal;
            expVal=l.expVal;
            return(*this);
        }
        Log &operator +=(const Log &l)
        {
            logVal=log_add(logVal,l.logVal);
            expVal=-1;
            return(*this);
        }
        Log &operator -=(const Log &l)
        {
            logVal=log_subtract(logVal,l.logVal);
            expVal=-1;
            return(*this);
        }
        Log &operator *=(const Log &l)
        {
            logVal=log_multiply(logVal,l.logVal);
            expVal=-1;
            return(*this);
        }
        Log &operator /=(const Log &l)
        {
            logVal=log_divide(logVal,l.logVal);
            expVal=-1;
            return(*this);
        }
        real_t exp()
        {
            if (expVal<0)
                expVal=safe_exp(logVal);
            return(expVal);
        }
        real_t log() const
        {
            return(logVal);
        }
    };
    Log operator +(const Log &log1,const Log &log2)
    {
        return(Log(Log::log_add(log1.log(),log2.log()),true));
    }
    Log operator -(const Log &log1,const Log &log2)
    {
        return(Log(Log::log_subtract(log1.log(),log2.log()),true));
    }
    Log operator *(const Log &log1,const Log &log2)
    {
        return(Log(Log::log_multiply(log1.log(),log2.log()),true));
    }
    Log operator /(const Log &log1,const Log &log2)
    {
        return(Log(Log::log_divide(log1.log(),log2.log()),true));
    }
    bool operator <(const Log &log1,const Log &log2)
    {
        return(log1.log()<log2.log());
    }
    bool operator >(const Log &log1,const Log &log2)
    {
        return(log1.log()>log2.log());
    }
    bool operator <=(const Log &log1,const Log &log2)
    {
        return(log1.log()<=log2.log());
    }
    bool operator >=(const Log &log1,const Log &log2)
    {
        return(log1.log()>=log2.log());
    }
    bool operator ==(const Log &log1,const Log &log2)
    {
        return(log1.log()==log2.log());
    }
    bool operator !=(const Log &log1,const Log &log2)
    {
        return(log1.log()!=log2.log());
    }
    std::istream &operator >>(std::istream &in,Log &l)
    {
        real_t d;
        in>>d;
        l=Log(d,true);
        return(in);
    }
    std::ostream &operator <<(std::ostream &out,const Log &l)
    {
        out<<l.log();
        return(out);
    }
    real_t Log::expMax=std::numeric_limits<real_t>::max();
    real_t Log::expMin=std::numeric_limits<real_t>::min();
    real_t Log::expLimit=std::log(expMax);
    real_t Log::logInfinity=1e38;
    real_t Log::logZero=-Log::logInfinity;
}
}

namespace layers
{

    template <typename TDevice> CtcLayer <TDevice>::CtcLayer(const helpers::JsonValue &layerChild,TrainableLayer <TDevice> &precedingLayer)
        :PostOutputLayer <TDevice>(layerChild,precedingLayer)
    {
    }

    template <typename TDevice> CtcLayer <TDevice>::~CtcLayer()
    {
    }

    template <typename TDevice> real_t CtcLayer <TDevice>::calculateError()
    {
        output.resize(this->_actualOutputs().size());
        thrust::copy(this->_actualOutputs().begin(),this->_actualOutputs().end(),output.begin());
        outputErrors.resize(this->_outputErrors().size());
        thrust::fill(outputErrors.begin(),outputErrors.end(),0);
        real_t error=0;
        for (int i=0;i<realSeqNum;i++)
        {
            error+=calculateError(i);
            targetLabel[i].clear();
        }
        delete[] targetLabel;
        return(error);
    }

    template <typename TDevice> real_t CtcLayer <TDevice>::calculateError(int id)
    {

        totalTime=targetLabel[id].size();
        targetLabel[id].erase(thrust::find(targetLabel[id].begin(),targetLabel[id].end(),-1),targetLabel[id].end());
        const int_vector &targetLabelSeq=targetLabel[id];
        typedef thrust::host_vector <internal::Log> ctc_vector;
        int blank=this->size()-1;
        totalSegments=targetLabelSeq.size()*2+1;

        //calculate the forward variables
        ctc_vector *forwardVariables=new ctc_vector[totalTime];
        forwardVariables[0].resize(totalSegments);
        forwardVariables[0][0]=activation(id,0,blank);
        if (totalSegments>1)
            forwardVariables[0][1]=activation(id,0,targetLabelSeq[0]);
        for (int t=1;t<totalTime;t++)
        {
            const ctc_vector &oldFvars=forwardVariables[t-1];
            ctc_vector &fvars=forwardVariables[t];
            fvars.resize(totalSegments);
            thrust::pair <int,int> bound=segment_range(t);
            for (int s=bound.first;s<bound.second;s++)
            {
                internal::Log fv;
                if (s&1)
                {
                    int labelIndex=s/2;
                    int labelNum=targetLabelSeq[labelIndex];
                    fv=oldFvars[s]+oldFvars[s-1];
                    if (s>1 && labelNum!=targetLabelSeq[labelIndex-1])
                        fv+=oldFvars[s-2];
                    fv*=activation(id,t,labelNum);
                }
                else
                {
                    fv=oldFvars[s];
                    if (s)
                        fv+=oldFvars[s-1];
                    fv*=activation(id,t,blank);
                }
                fvars[s]=fv;
            }
        }

        //calculate the backward vairables
        ctc_vector *backwardVariables=new ctc_vector[totalTime];
        ctc_vector &lastBvs=backwardVariables[totalTime-1];
        lastBvs.resize(totalSegments);
        lastBvs.back()=1;
        if (totalSegments>1)
           lastBvs[lastBvs.size()-2]=1;
        for (int t=totalTime-2;t>=0;t--)
        {
            const ctc_vector &oldBvars=backwardVariables[t+1];
            ctc_vector &bvars=backwardVariables[t];
            bvars.resize(totalSegments);
            thrust::pair <int,int> bound=segment_range(t);
            for (int s=bound.first;s<bound.second;s++)
            {
                internal::Log bv;
                if (s&1)
                {
                    int labelIndex=s/2;
                    int labelNum=targetLabelSeq[labelIndex];
                    bv=oldBvars[s]*activation(id,t+1,labelNum)+oldBvars[s+1]*activation(id,t+1,blank);
                    if (s<totalSegments-2)
                    {
                        int nextLabelNum=targetLabelSeq[labelIndex+1];
                        if (labelNum!=nextLabelNum)
                            bv+=oldBvars[s+2]*activation(id,t+1,nextLabelNum);
                    }
                }
                else
                {
                    bv=oldBvars[s]*activation(id,t+1,blank);
                    if (s<totalSegments-1)
                        bv+=oldBvars[s+1]*activation(id,t+1,targetLabelSeq[s/2]);
                }
                bvars[s]=bv;
            }
        }

        //calcuate the errors
        const ctc_vector &lastFvs=forwardVariables[totalTime-1];
        internal::Log logProb=lastFvs.back();
        if (totalSegments>1)
            logProb+=lastFvs[lastFvs.size()-2];

        if (logProb==0.0)
        {
            for (int i=0;i<totalTime;i++)
            {
                forwardVariables[i].clear();
                backwardVariables[i].clear();
            }
            delete[] forwardVariables;
            delete[] backwardVariables;
            return 0.0;
        }

        ctc_vector dEdYTerms;
        dEdYTerms.resize(this->size());
        for (int time=0;time<totalTime;time++)
        {
            thrust::fill(dEdYTerms.begin(),dEdYTerms.end(),0);
            const ctc_vector &fvars=forwardVariables[time];
            const ctc_vector &bvars=backwardVariables[time];
            for (int s=0;s<totalSegments;s++)
            {
                int k=s&1?targetLabelSeq[s/2]:blank;
                dEdYTerms[k]+=fvars[s]*bvars[s];
            }
            for (int i=0;i<this->size();i++)
            {
                internal::Log t=dEdYTerms[i]/(logProb*activation(id,time,i));
                outputErrors[(time*seqNum+id)*this->size()+i]=-(dEdYTerms[i]/(logProb*activation(id,time,i))).exp();
            }
        }

        for (int i=0;i<totalTime;i++)
        {
            forwardVariables[i].clear();
            backwardVariables[i].clear();
        }
        delete[] forwardVariables;
        delete[] backwardVariables;

        return(-logProb.log());
    }

    template <typename TDevice> real_t CtcLayer <TDevice>::activation(int id,int time,int offset)
    {
        return thrust::max(internal::Log::expMin,output[(time*seqNum+id)*this->size()+offset]);
    }

    template <typename TDevice> thrust::pair <int,int> CtcLayer <TDevice>::segment_range(int time) const
    {
        int start=thrust::max(0,totalSegments-(2*(totalTime-time)));
        int end=thrust::min(totalSegments,2*(time+1));
        return(thrust::make_pair(start,end));
    }

    template <typename TDevice> const std::string &CtcLayer <TDevice>::type() const
    {
        static std::string s="connectionist_temporal_classification";
        return(s);
    }

    template <typename TDevice> void CtcLayer <TDevice>::loadSequences(const data_sets::DataSetFraction &fraction)
    {

        PostOutputLayer <TDevice>::loadSequences(fraction);

        seqNum=fraction.targetClasses().size()/fraction.maxSeqLength();
        realSeqNum=fraction.numSequences();
        targetLabel=new int_vector[realSeqNum];

        for (int i=0;i<realSeqNum;i++)
        {
            targetLabel[i].resize(fraction.seqInfo(i).length);
            for (int j=0;j<targetLabel[i].size();j++)
                targetLabel[i][j]=fraction.targetClasses()[j*seqNum+i];
        }
    }

    template <typename TDevice> void CtcLayer <TDevice>::computeForwardPass()
    {
    }

    template <typename TDevice> void CtcLayer <TDevice>::computeBackwardPass()
    {
        thrust::copy(outputErrors.begin(),outputErrors.end(),this->_outputErrors().begin());
    }

    template class CtcLayer <Cpu>;
    template class CtcLayer <Gpu>;

}
