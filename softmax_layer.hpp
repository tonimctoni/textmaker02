#include "matrix.hpp"
#include <memory>


#ifndef __SOFTMAXLAYER__
#define __SOFTMAXLAYER__
template<unsigned long input_size, unsigned long output_size, unsigned long batch_size, unsigned long time_steps>
class SoftmaxLayerBase
{
private:
    std::unique_ptr<std::array<Matrix<batch_size,output_size>,time_steps>> outputs;
    std::unique_ptr<std::array<Matrix<batch_size,output_size>,time_steps>> output_deltas;
    std::unique_ptr<Matrix<input_size, output_size>> weights;
    std::unique_ptr<Matrix<1, output_size>> bias;
public:
    SoftmaxLayerBase():
    outputs(new std::array<Matrix<batch_size,output_size>,time_steps>),
    output_deltas(new std::array<Matrix<batch_size,output_size>,time_steps>),
    weights(new Matrix<input_size, output_size>),
    bias(new Matrix<1, output_size>)
    {
        weights->randomize_for_nn(input_size+1);
        bias->randomize_for_nn(input_size+1);
    }

    void calc(const Matrix<batch_size,input_size> &X, size_t time_step)
    {
        assert(time_step<time_steps);
        (*outputs)[time_step].equals_a_dot_b(X, *weights);
        (*outputs)[time_step].add_to_each_row(*bias);
        (*outputs)[time_step].apply_softmax();
    }

    inline void set_first_delta_and_propagate_with_cross_enthropy(const Matrix<batch_size,output_size> &Y, Matrix<batch_size,input_size> &X_delta, size_t time_step)
    {
        assert(time_step<time_steps);
        (*output_deltas)[time_step].equals_a_sub_b(Y,(*outputs)[time_step]);
        X_delta.equals_a_dot_bt((*output_deltas)[time_step], *weights);
    }

    inline void set_first_delta(const Matrix<batch_size,output_size> &Y, size_t time_step)
    {
        assert(time_step<time_steps);
        (*output_deltas)[time_step].equals_a_sub_b(Y,(*outputs)[time_step]);
    }

    inline void propagate_delta(size_t time_step)
    {
        assert(time_step<time_steps);
        (*output_deltas)[time_step].mult_after_func03((*outputs)[time_step]);
    }

    inline void propagate_delta(Matrix<batch_size,input_size> &X_delta, size_t time_step)
    {
        propagate_delta(time_step);
        X_delta.equals_a_dot_bt((*output_deltas)[time_step], *weights);
    }

    inline const Matrix<batch_size,output_size>& get_output(size_t time_step) const noexcept
    {
        return (*outputs)[time_step];
    }

    inline Matrix<batch_size,output_size>& get_delta_output(size_t time_step) noexcept
    {
        return (*output_deltas)[time_step];
    }

    inline void update_weights_without_optimizer(const Matrix<batch_size,input_size> &X, size_t time_step, double learning_rate)
    {
        assert(time_step<time_steps);
        (*weights).add_factor_mul_at_dot_b(learning_rate, X, (*output_deltas)[time_step]);
        (*bias).add_factor_mul_each_row_of_a(learning_rate, (*output_deltas)[time_step]);
    }
};
#endif