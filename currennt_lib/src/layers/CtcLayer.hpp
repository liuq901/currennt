#ifndef LAYERS_CTCLAYER_HPP
#define LAYERS_CTCLAYER_HPP

#include "PostOutputLayer.hpp"

namespace layers {
	template <typename TDevice>
	class CtcLayer : public PostOutputLayer <TDevice>
	{
        typedef Cpu::int_vector int_vector;
        typedef Cpu::real_vector real_vector;

    private:
        real_vector outputErrors,output;
        int_vector *targetLabel;
        int seqNum,realSeqNum,totalTime,totalSegments;
        
        real_t activation(int,int,int);

        thrust::pair <int,int> segment_range(int) const;

        real_t calculateError(int);

    public:
        CtcLayer(const helpers::JsonValue &layerChild,TrainableLayer<TDevice> &precedingLayer);

        virtual ~CtcLayer();

        virtual real_t calculateError();

        virtual const std::string &type() const;

        virtual void loadSequences(const data_sets::DataSetFraction &fraction);

        virtual void computeForwardPass();

        virtual void computeBackwardPass();
	};
}

#endif
