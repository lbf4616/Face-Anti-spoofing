//
// MATLAB Compiler: 6.4 (R2017a)
// Date: Tue Jul 03 11:06:28 2018
// Arguments: "-B""macro_default""-W""cpplib:spdiags""-T""link:lib""spdiags"
//

#include <stdio.h>
#define EXPORTING_spdiags 1
#include "spdiags.h"

static HMCRINSTANCE _mcr_inst = NULL;


#if defined( _MSC_VER) || defined(__BORLANDC__) || defined(__WATCOMC__) || defined(__LCC__) || defined(__MINGW64__)
#ifdef __LCC__
#undef EXTERN_C
#endif
#include <windows.h>

static char path_to_dll[_MAX_PATH];

BOOL WINAPI DllMain(HINSTANCE hInstance, DWORD dwReason, void *pv)
{
    if (dwReason == DLL_PROCESS_ATTACH)
    {
        if (GetModuleFileName(hInstance, path_to_dll, _MAX_PATH) == 0)
            return FALSE;
    }
    else if (dwReason == DLL_PROCESS_DETACH)
    {
    }
    return TRUE;
}
#endif
#ifdef __cplusplus
extern "C" {
#endif

static int mclDefaultPrintHandler(const char *s)
{
  return mclWrite(1 /* stdout */, s, sizeof(char)*strlen(s));
}

#ifdef __cplusplus
} /* End extern "C" block */
#endif

#ifdef __cplusplus
extern "C" {
#endif

static int mclDefaultErrorHandler(const char *s)
{
  int written = 0;
  size_t len = 0;
  len = strlen(s);
  written = mclWrite(2 /* stderr */, s, sizeof(char)*len);
  if (len > 0 && s[ len-1 ] != '\n')
    written += mclWrite(2 /* stderr */, "\n", sizeof(char));
  return written;
}

#ifdef __cplusplus
} /* End extern "C" block */
#endif

/* This symbol is defined in shared libraries. Define it here
 * (to nothing) in case this isn't a shared library. 
 */
#ifndef LIB_spdiags_C_API
#define LIB_spdiags_C_API /* No special import/export declaration */
#endif

LIB_spdiags_C_API 
bool MW_CALL_CONV spdiagsInitializeWithHandlers(
    mclOutputHandlerFcn error_handler,
    mclOutputHandlerFcn print_handler)
{
    int bResult = 0;
  if (_mcr_inst != NULL)
    return true;
  if (!mclmcrInitialize())
    return false;
  if (!GetModuleFileName(GetModuleHandle("spdiags"), path_to_dll, _MAX_PATH))
    return false;
    {
        mclCtfStream ctfStream = 
            mclGetEmbeddedCtfStream(path_to_dll);
        if (ctfStream) {
            bResult = mclInitializeComponentInstanceEmbedded(   &_mcr_inst,
                                                                error_handler, 
                                                                print_handler,
                                                                ctfStream);
            mclDestroyStream(ctfStream);
        } else {
            bResult = 0;
        }
    }  
    if (!bResult)
    return false;
  return true;
}

LIB_spdiags_C_API 
bool MW_CALL_CONV spdiagsInitialize(void)
{
  return spdiagsInitializeWithHandlers(mclDefaultErrorHandler, mclDefaultPrintHandler);
}

LIB_spdiags_C_API 
void MW_CALL_CONV spdiagsTerminate(void)
{
  if (_mcr_inst != NULL)
    mclTerminateInstance(&_mcr_inst);
}

LIB_spdiags_C_API 
void MW_CALL_CONV spdiagsPrintStackTrace(void) 
{
  char** stackTrace;
  int stackDepth = mclGetStackTrace(&stackTrace);
  int i;
  for(i=0; i<stackDepth; i++)
  {
    mclWrite(2 /* stderr */, stackTrace[i], sizeof(char)*strlen(stackTrace[i]));
    mclWrite(2 /* stderr */, "\n", sizeof(char)*strlen("\n"));
  }
  mclFreeStackTrace(&stackTrace, stackDepth);
}


LIB_spdiags_C_API 
bool MW_CALL_CONV mlxSpdiags(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "spdiags", nlhs, plhs, nrhs, prhs);
}

LIB_spdiags_CPP_API 
void MW_CALL_CONV spdiags(int nargout, mwArray& res1, mwArray& res2, const mwArray& arg1, 
                          const mwArray& arg2, const mwArray& arg3, const mwArray& arg4)
{
  mclcppMlfFeval(_mcr_inst, "spdiags", nargout, 2, 4, &res1, &res2, &arg1, &arg2, &arg3, &arg4);
}

