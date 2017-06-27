#include "matrix.hpp"
#include <memory>


#ifndef __TANHLAYER__
#define __TANHLAYER__
template<unsigned long input_size, unsigned long output_size, unsigned long batch_size, unsigned long time_steps>
class TanhLayerBase
{
protected:
    std::array<Matrix<batch_size,output_size>,time_steps> outputs;
    std::array<Matrix<batch_size,output_size>,time_steps> output_deltas;
    Matrix<input_size, output_size> weights;
    Matrix<1, output_size> bias;
public:
    TanhLayerBase() noexcept
    {
        weights.randomize_for_nn(input_size+1);
        bias.randomize_for_nn(input_size+1);
    }

    void show_guts() const noexcept
    {
        print("TanhLayer", input_size, output_size, batch_size, time_steps);
        print("weights:");
        print(weights);
        print("bias");
        print(bias);
    }

    bool has_nan() const noexcept
    {
        return weights.has_nan() or bias.has_nan();
    }

    bool has_inf() const noexcept
    {
        return weights.has_inf() or bias.has_inf();
    }

    const Matrix<input_size, output_size>& get_weights() const noexcept {return weights;}
    const Matrix<1, output_size>& get_bias() const noexcept {return bias;}

    template<unsigned long other_batch_size, unsigned long other_time_steps>
    void set_wb(const TanhLayerBase<input_size, output_size, other_batch_size, other_time_steps> &other)
    {
        weights.set(other.get_weights());
        bias.set(other.get_bias());
    }

    void calc(const Matrix<batch_size,input_size> &X, size_t time_step)
    {
        assert(time_step<time_steps);
        outputs[time_step].equals_a_dot_b(X, weights);
        outputs[time_step].add_to_each_row(bias);
        outputs[time_step].apply_tanh();
    }

    inline void set_first_delta(const Matrix<batch_size,output_size> &Y, size_t time_step)
    {
        assert(time_step<time_steps);
        output_deltas[time_step].equals_a_sub_b(Y,outputs[time_step]);
    }

    inline void propagate_delta(size_t time_step)
    {
        assert(time_step<time_steps);
        output_deltas[time_step].mult_after_func02(outputs[time_step]);
    }

    inline void propagate_delta(Matrix<batch_size,input_size> &X_delta, size_t time_step)
    {
        propagate_delta(time_step);
        X_delta.equals_a_dot_bt(output_deltas[time_step], weights);
    }

    inline const Matrix<batch_size,output_size>& get_output(size_t time_step) const noexcept
    {
        return outputs[time_step];
    }

    inline Matrix<batch_size,output_size>& get_delta_output(size_t time_step) noexcept
    {
        return output_deltas[time_step];
    }

    inline void update_weights_without_optimizer(const Matrix<batch_size,input_size> &X, size_t time_step, double learning_rate)
    {
        assert(time_step<time_steps);
        weights.add_factor_mul_at_dot_b(learning_rate, X, output_deltas[time_step]);
        bias.add_factor_mul_each_row_of_a(learning_rate, output_deltas[time_step]);
    }
};

template<unsigned long input_size, unsigned long output_size, unsigned long batch_size, unsigned long time_steps>
class TanhLayerRMSProp: public TanhLayerBase<input_size, output_size, batch_size, time_steps>
{
private:
    Matrix<input_size, output_size> ms_weights;
    Matrix<1, output_size> ms_bias;
public:
    using TanhLayerBase<input_size, output_size, batch_size, time_steps>::output_deltas;
    using TanhLayerBase<input_size, output_size, batch_size, time_steps>::weights;
    using TanhLayerBase<input_size, output_size, batch_size, time_steps>::bias;
    TanhLayerRMSProp()noexcept:TanhLayerBase<input_size, output_size, batch_size, time_steps>(), ms_weights(1.0), ms_bias(1.0)
    {
    }

    inline void update_weights_with_rmsprop(const Matrix<batch_size,input_size> &X, size_t time_step, const double learning_rate, const double decay) noexcept
    {
        assert(time_step<time_steps);
        update_weights_and_ms_with_rmsprop(X, output_deltas[time_step], weights, ms_weights, learning_rate, decay);
        update_bias_and_ms_with_rmsprop(output_deltas[time_step], bias, ms_bias, learning_rate, decay);
    }
};
#endif