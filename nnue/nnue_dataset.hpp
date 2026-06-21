// nnue/nnue_dataset.hpp
#pragma once
#include "replay_buffer.hpp"

namespace NNUE {
    // Redirects any old code using 'NNUEDataset' to point to 'ReplayBuffer'
    using NNUEDataset = ReplayBuffer;
}
