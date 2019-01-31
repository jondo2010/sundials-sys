#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(test)]
mod tests {
    use crate::*;
    use core::ffi::c_void;
    use std::slice;

    #[test]
    fn test_ida() {
        unsafe extern "C" fn resrob(
            tres: realtype,
            yy: N_Vector,
            yp: N_Vector,
            rr: N_Vector,
            user_data: *mut c_void,
        ) -> i32 {
            let yval = N_VGetArrayPointer(yy);
            let ypval = N_VGetArrayPointer(yp);
            let rval = N_VGetArrayPointer(rr);

            *rval.offset(0) =
                (-0.04) * (*yval.offset(0)) + (1.0e4) * (*yval.offset(1)) * (*yval.offset(2));
            *rval.offset(1) = -(*rval.offset(0))
                - (3.0e7) * (*yval.offset(1)) * (*yval.offset(1))
                - (*ypval.offset(1));
            *rval.offset(0) -= *ypval.offset(0);
            *rval.offset(2) = *yval.offset(0) + *yval.offset(1) + *yval.offset(2) - 1.0;

            return 0;
        }

        const NEQ: i64 = 3;
        const NOUT: i64 = 12;

        unsafe {
            let yy = N_VNew_Serial(NEQ);
            let yp = N_VNew_Serial(NEQ);

            /* Call IDACreate and IDAInit to initialize IDA memory */
            let mut mem = IDACreate();
            let retval = IDAInit(mem, Some(resrob), 0.0, yy, yp);

            /* Call IDASVtolerances to set tolerances */
            IDASStolerances(mem, 1e-4, 1e-6);
            //retval = IDASVtolerances(mem, 1e-4, avtol);

            /* Call IDARootInit to specify the root function grob with 2 components */
            //retval = IDARootInit(mem, 2, grob);

            /* Create sparse SUNMatrix for use in linear solves */
            let nnz = NEQ * NEQ;
            let A = SUNSparseMatrix(NEQ, NEQ, nnz, CSC_MAT as std::os::raw::c_int);

            /* Create SuperLUMT SUNLinearSolver object (one thread) */
            //let LS = SUNLinSol_SuperLUMT(yy, A, 1);

            /* Attach the matrix and linear solver */
            //let retval = IDASetLinearSolver(mem, LS, A);

            /* Set the user-supplied Jacobian routine */
            //let retval = IDASetJacFn(mem, jacrob);

            // In loop, call IDASolve, print results, and test for error. Break out of loop when NOUT preset output times have been reached.

            let mut iout = 0;
            let mut tout = 0.4;
            let mut tret = 0.0;
            let mut rootsfound: [i32; 2] = [0, 0];
            loop {
                let retval = IDASolve(mem, tout, &mut tret, yy, yp, IDA_NORMAL);
                if (retval == IDA_ROOT_RETURN) {
                    let retvalr = IDAGetRootInfo(mem, rootsfound.as_mut_ptr());
                    //PrintRootInfo(rootsfound[0],rootsfound[1]);
                }

                if (retval == IDA_SUCCESS) {
                    iout += 1;
                    tout *= 10.0;
                }

                if iout == NOUT {
                    break;
                };
            }
        }
    }

    #[test]
    // This just tests if the most basic of all programs works. More tests to come soon.
    fn simple_ode() {
        unsafe extern "C" fn rhs(
            _t: realtype,
            y: N_Vector,
            dy: N_Vector,
            _user_data: *mut c_void,
        ) -> i32 {
            *N_VGetArrayPointer(dy) = -*N_VGetArrayPointer(y);
            return 0;
        }

        unsafe {
            let y = N_VNew_Serial(1);
            *N_VGetArrayPointer(y) = 1.0;

            let mut cvode_mem = CVodeCreate(CV_ADAMS);

            CVodeInit(cvode_mem, Some(rhs), 0.0, y);
            CVodeSStolerances(cvode_mem, 1e-6, 1e-8);

            let matrix = SUNDenseMatrix(1, 1);
            let solver = SUNDenseLinearSolver(y, matrix);

            CVodeSetLinearSolver(cvode_mem, solver, matrix);

            let mut t = 0f64;
            CVode(cvode_mem, 1.0, y, &mut t, CV_NORMAL);
            // y[0] is now exp(-1)

            let result = (*N_VGetArrayPointer(y) * 1e6) as i32;
            assert_eq!(result, 367879);

            N_VDestroy(y);
            CVodeFree(&mut cvode_mem);
            SUNLinSolFree(solver);
            SUNMatDestroy(matrix);
        }
    }
}
