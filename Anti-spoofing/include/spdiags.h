//
// MATLAB Compiler: 6.4 (R2017a)
// Date: Tue Jul 03 11:06:28 2018
// Arguments: "-B""macro_default""-W""cpplib:spdiags""-T""link:lib""spdiags"
//

#ifndef __spdiags_h
#define __spdiags_h 1

#if defined(__cplusplus) && !defined(mclmcrrt_h) && defined(__linux__)
#  pragma implementation "mclmcrrt.h"
#endif
#include "mclmcrrt.h"
#include "mclcppclass.h"
#ifdef __cplusplus
extern "C" {
#endif

#if defined(__SUNPRO_CC)
/* Solaris shared libraries use __global, rather than mapfiles
 * to define the API exported from a shared library. __global is
 * only necessary when building the library -- files including
 * this header file to use the library do not need the __global
 * declaration; hence the EXPORTING_<library> logic.
 */

#ifdef EXPORTING_spdiags
#define PUBLIC_spdiags_C_API __global
#else
#define PUBLIC_spdiags_C_API /* No import statement needed. */
#endif

#define LIB_spdiags_C_API PUBLIC_spdiags_C_API

#elif defined(_HPUX_SOURCE)

#ifdef EXPORTING_spdiags
#define PUBLIC_spdiags_C_API __declspec(dllexport)
#else
#define PUBLIC_spdiags_C_API __declspec(dllimport)
#endif

#define LIB_spdiags_C_API PUBLIC_spdiags_C_API


#else

#define LIB_spdiags_C_API

#endif

/* This symbol is defined in shared libraries. Define it here
 * (to nothing) in case this isn't a shared library. 
 */
#ifndef LIB_spdiags_C_API 
#define LIB_spdiags_C_API /* No special import/export declaration */
#endif

extern LIB_spdiags_C_API 
bool MW_CALL_CONV spdiagsInitializeWithHandlers(
       mclOutputHandlerFcn error_handler, 
       mclOutputHandlerFcn print_handler);

extern LIB_spdiags_C_API 
bool MW_CALL_CONV spdiagsInitialize(void);

extern LIB_spdiags_C_API 
void MW_CALL_CONV spdiagsTerminate(void);



extern LIB_spdiags_C_API 
void MW_CALL_CONV spdiagsPrintStackTrace(void);

extern LIB_spdiags_C_API 
bool MW_CALL_CONV mlxSpdiags(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);


#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

/* On Windows, use __declspec to control the exported API */
#if defined(_MSC_VER) || defined(__BORLANDC__)

#ifdef EXPORTING_spdiags
#define PUBLIC_spdiags_CPP_API __declspec(dllexport)
#else
#define PUBLIC_spdiags_CPP_API __declspec(dllimport)
#endif

#define LIB_spdiags_CPP_API PUBLIC_spdiags_CPP_API

#else

#if !defined(LIB_spdiags_CPP_API)
#if defined(LIB_spdiags_C_API)
#define LIB_spdiags_CPP_API LIB_spdiags_C_API
#else
#define LIB_spdiags_CPP_API /* empty! */ 
#endif
#endif

#endif

extern LIB_spdiags_CPP_API void MW_CALL_CONV spdiags(int nargout, mwArray& res1, mwArray& res2, const mwArray& arg1, const mwArray& arg2, const mwArray& arg3, const mwArray& arg4);

#endif
#endif
