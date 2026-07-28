#pragma once

#include "ggml.h"

namespace lamio {

// Eval callback for the ggml backend scheduler.
// Intercepts mul_mat_id nodes to lazily load MoE expert weights.
//
// t:    the current graph node
// ask:  true = before compute, false = after compute
// user_data: opaque pointer (unused for now)
//
// Returns true to continue compute, false to abort.
bool lamio_eval_callback(ggml_tensor * t, bool ask, void * user_data);

} // namespace lamio