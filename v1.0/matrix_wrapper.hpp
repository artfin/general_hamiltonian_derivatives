#pragma once

#include <vector>
#include <functional>
#include <Eigen/Dense>

template <int _Rows, int _Cols>
class MatrixWrapper
{
public:
    typedef std::function<void(Eigen::Ref<Eigen::Matrix<double, _Rows, _Cols>>,
                               std::vector<QData> const&)> filler_type;

    explicit MatrixWrapper() 
    { 
        m = Eigen::Matrix<double, _Rows, _Cols>::Zero( _Rows, _Cols );
    }

    MatrixWrapper( filler_type filler_ ) 
    {
        m = Eigen::Matrix<double, _Rows, _Cols>::Zero( _Rows, _Cols );
        filler = filler_;
    }
    
    void set_filler( filler_type filler_ )
    {
        filler = filler_;
    }

    void fill( std::vector<QData> q ) 
    {
        filler( m, q );
    }

    const Eigen::Ref<const Eigen::Matrix<double, _Rows, _Cols>> get() const {
        return m;
    }

private:
    Eigen::Matrix<double, _Rows, _Cols> m;
    filler_type filler;
};


