#pragma once

#include "hamiltonian_derivatives.hpp"
#include "matrix_wrapper.hpp"
#include "generated.hpp"

#include "co2_ar/ai_pes_co2ar.hpp"

namespace co2_ar {
    HamiltonianDerivatives<Q_SIZE> create_derivatives_object()
    {
        HamiltonianDerivatives<Q_SIZE> ham;

        ham.set_IT_filler( fill_IT );
        ham.set_A_filler( fill_A );
        ham.set_a_filler( fill_a );

        std::vector< HamiltonianDerivatives<Q_SIZE>::IT_filler_type > IT_derivatives_fillers = {
            fill_IT_dR,
            fill_IT_dTheta
        };

        std::vector< HamiltonianDerivatives<Q_SIZE>::A_filler_type > A_derivatives_fillers = {
            fill_A_dR,
            fill_A_dTheta
        };

        std::vector< HamiltonianDerivatives<Q_SIZE>::a_filler_type > a_derivatives_fillers = {
            fill_a_dR,
            fill_a_dTheta
        };

        ham.set_IT_derivatives_fillers( IT_derivatives_fillers );
        ham.set_A_derivatives_fillers( A_derivatives_fillers );
        ham.set_a_derivatives_fillers( a_derivatives_fillers );
        
        AI_PES_co2_ar pes;
        ham.set_potential( pes.pes_q, pes.pes_derivatives ); 

        return ham;
    }
}
