#ifndef __MYMATRIX78256__IAMTONI__
#define __MYMATRIX78256__IAMTONI__
#include <array>
#include <iostream>
#include <cmath>
#include <random>
#include <fstream>
#include "mystuff.hpp"

template<unsigned long M, unsigned long N>
class Matrix: public std::array<std::array<double,N>,M>
{
private:
public:
    Matrix()=default;

    Matrix(double s) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]=s;
    }

    inline void set(const Matrix &rhs) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]=rhs[i][j];
    }

    inline void add(const Matrix& rhs) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]+=rhs[i][j];
    }

    template<unsigned long L>
    inline void equals_a_dot_b(const Matrix<M,L>& a, const Matrix<L,N>& b) noexcept
    {
        #pragma omp parallel for collapse(1) default(shared)
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
            {
                double acc=0.0;
                for(size_t k=0;k<L;k++) acc+=a[i][k]*b[k][j];
                (*this)[i][j]=acc;
            }
    }

    template<unsigned long L>
    inline void equals_a_dot_bt(const Matrix<M,L>& a, const Matrix<N,L>& b) noexcept
    {
        #pragma omp parallel for collapse(1) default(shared)
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
            {
                double acc=0.0;
                for(size_t k=0;k<L;k++) acc+=a[i][k]*b[j][k];
                (*this)[i][j]=acc;
            }
    }

    template<unsigned long L>
    inline void add_a_dot_b(const Matrix<M,L>& a, const Matrix<L,N>& b) noexcept
    {
        #pragma omp parallel for collapse(1) default(shared)
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
            {
                double acc=0.0;
                for(size_t k=0;k<L;k++) acc+=a[i][k]*b[k][j];
                (*this)[i][j]+=acc;
            }
    }

    template<unsigned long L>
    inline void add_a_dot_bt(const Matrix<M,L>& a, const Matrix<N,L>& b) noexcept
    {
        #pragma omp parallel for collapse(1) default(shared)
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
            {
                double acc=0.0;
                for(size_t k=0;k<L;k++) acc+=a[i][k]*b[j][k];
                (*this)[i][j]+=acc;
            }
    }

    inline void equals_a_sub_b(const Matrix& a, const Matrix& b) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]=a[i][j]-b[i][j];
    }

    inline void apply_sigmoid() noexcept
    {
        #pragma omp parallel for collapse(1) default(shared)
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]=1.0/(1.0+std::exp(-(*this)[i][j]));
    }

    inline void apply_tanh() noexcept
    {
        #pragma omp parallel for collapse(1) default(shared)
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]=std::tanh((*this)[i][j]);
    }

    inline void apply_softmax() noexcept
    {
        double sum=0.0;
        #pragma omp parallel for collapse(1) default(shared)
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]=std::exp((*this)[i][j]);
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                sum+=(*this)[i][j];
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]/=sum;
    }

    inline void mult_after_func01(const Matrix &a) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]*=(a[i][j])*(1.0-a[i][j]);
    }

    inline void mult_after_func02(const Matrix &a) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]*=1-(a[i][j])*(a[i][j]);
    }

    inline void mult_after_func03(const Matrix &a) noexcept //Same as 01, but it will be used for softmax
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]*=(a[i][j])*(1.0-a[i][j]);
    }

    inline void randomize_for_nn(std::normal_distribution<double>::result_type scal) noexcept
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dst(0,1.0/(sqrt(scal)));
        for(auto &row:*this)
            for(auto &element:row)
                element=dst(gen);
    }

    inline void equals_a_mul_b_add_c_mul_d(const Matrix<M,N>& a, const Matrix<M,N>& b, const Matrix<M,N>& c, const Matrix<M,N>& d) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]=a[i][j]*b[i][j]+c[i][j]*d[i][j];
    }

    inline void equals_a_mul_b(const Matrix<M,N>& a, const Matrix<M,N>& b) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]=a[i][j]*b[i][j];
    }

    inline void add_to_each_row(const Matrix<1, N> &a)
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]+=a[0][j];
    }

    template<unsigned long A>
    inline void add_factor_mul_each_row_of_a(const double factor, const Matrix<A, N> &a)
    {
        static_assert(M==1, "M is not 1");
        for(size_t j=0;j<N;j++)
        {
            double sum=0.0;
            for(size_t i=0;i<M;i++)
                sum+=a[i][j];
            (*this)[0][j]+=factor*sum;
        }
    }

    template<unsigned long L>
    inline void add_factor_mul_at_dot_b(const double factor, const Matrix<L,M>& a, const Matrix<L,N>& b) noexcept
    {
        #pragma omp parallel for collapse(1) default(shared)
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
            {
                double acc=0.0;
                for(size_t k=0;k<L;k++) acc+=a[k][i]*b[k][j];
                (*this)[i][j]+=factor*acc;
            }
    }
};

template<unsigned long M, unsigned long N>
class OneHots
{
private:
    std::array<size_t,M> hot_index;
    Matrix<M, N> X;
public:
    OneHots()noexcept: X(0.0)
    {
        hot_index.fill(0);
    }

    inline void set(size_t indexindex, size_t index)
    {
        assert(indexindex<M);
        assert(index<N);
        X[indexindex][hot_index[indexindex]]=0.0;
        hot_index[indexindex]=index;
        X[indexindex][hot_index[indexindex]]=1.0;
    }

    inline void reset() noexcept
    {
        for(size_t i=0;i<M;i++)
            X[i][hot_index[i]]=0.0;
    }

    inline const Matrix<M,N>& get() const noexcept
    {
        return X;
    }
};

#endif