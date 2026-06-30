// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_GET_SREG_H_
#define _CUDA_PTX_GENERATED_GET_SREG_H_

/*
// mov.u32 sreg_value, %%tid.x; // PTX ISA 20
template <typename = void>
__device__ static inline uint32_t get_sreg_tid_x();
*/
#if __cccl_ptx_isa >= 200
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_tid_x()
{
  _CUDA_VSTD::uint32_t __sreg_value;
  asm("mov.u32 %0, %%tid.x;" : "=r"(__sreg_value) : :);
  return __sreg_value;
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u32 sreg_value, %%tid.y; // PTX ISA 20
template <typename = void>
__device__ static inline uint32_t get_sreg_tid_y();
*/
#if __cccl_ptx_isa >= 200
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_tid_y()
{
  _CUDA_VSTD::uint32_t __sreg_value;
  asm("mov.u32 %0, %%tid.y;" : "=r"(__sreg_value) : :);
  return __sreg_value;
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u32 sreg_value, %%tid.z; // PTX ISA 20
template <typename = void>
__device__ static inline uint32_t get_sreg_tid_z();
*/
#if __cccl_ptx_isa >= 200
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_tid_z()
{
  _CUDA_VSTD::uint32_t __sreg_value;
  asm("mov.u32 %0, %%tid.z;" : "=r"(__sreg_value) : :);
  return __sreg_value;
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u32 sreg_value, %%ntid.x; // PTX ISA 20
template <typename = void>
__device__ static inline uint32_t get_sreg_ntid_x();
*/
#if __cccl_ptx_isa >= 200
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_ntid_x()
{
  _CUDA_VSTD::uint32_t __sreg_value;
  asm volatile("mov.u32 %0, %%ntid.x;" : "=r"(__sreg_value) : :);
  return __sreg_value;
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u32 sreg_value, %%ntid.y; // PTX ISA 20
template <typename = void>
__device__ static inline uint32_t get_sreg_ntid_y();
*/
#if __cccl_ptx_isa >= 200
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_ntid_y()
{
  _CUDA_VSTD::uint32_t __sreg_value;
  asm volatile("mov.u32 %0, %%ntid.y;" : "=r"(__sreg_value) : :);
  return __sreg_value;
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u32 sreg_value, %%ntid.z; // PTX ISA 20
template <typename = void>
__device__ static inline uint32_t get_sreg_ntid_z();
*/
#if __cccl_ptx_isa >= 200
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_ntid_z()
{
  _CUDA_VSTD::uint32_t __sreg_value;
  asm volatile("mov.u32 %0, %%ntid.z;" : "=r"(__sreg_value) : :);
  return __sreg_value;
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u32 sreg_value, %%laneid; // PTX ISA 13
template <typename = void>
__device__ static inline uint32_t get_sreg_laneid();
*/
#if __cccl_ptx_isa >= 130
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_laneid()
{
  _CUDA_VSTD::uint32_t __sreg_value;
  asm("mov.u32 %0, %%laneid;" : "=r"(__sreg_value) : :);
  return __sreg_value;
}
#endif // __cccl_ptx_isa >= 130

/*
// mov.u32 sreg_value, %%warpid; // PTX ISA 13
template <typename = void>
__device__ static inline uint32_t get_sreg_warpid();
*/
#if __cccl_ptx_isa >= 130
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_warpid()
{
  _CUDA_VSTD::uint32_t __sreg_value;
  asm volatile("mov.u32 %0, %%warpid;" : "=r"(__sreg_value) : :);
  return __sreg_value;
}
#endif // __cccl_ptx_isa >= 130

/*
// mov.u32 sreg_value, %%nwarpid; // PTX ISA 20, SM_35
template <typename = void>
__device__ static inline uint32_t get_sreg_nwarpid();
*/
#if __cccl_ptx_isa >= 200
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_nwarpid_is_not_supported_before_SM_35__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_nwarpid()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 350
  _CUDA_VSTD::uint32_t __sreg_value;
  asm volatile("mov.u32 %0, %%nwarpid;" : "=r"(__sreg_value) : :);
  return __sreg_value;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_get_sreg_nwarpid_is_not_supported_before_SM_35__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u32 sreg_value, %%ctaid.x; // PTX ISA 20
template <typename = void>
__device__ static inline uint32_t get_sreg_ctaid_x();
*/
#if __cccl_ptx_isa >= 200
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_ctaid_x()
{
  _CUDA_VSTD::uint32_t __sreg_value;
  asm("mov.u32 %0, %%ctaid.x;" : "=r"(__sreg_value) : :);
  return __sreg_value;
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u32 sreg_value, %%ctaid.y; // PTX ISA 20
template <typename = void>
__device__ static inline uint32_t get_sreg_ctaid_y();
*/
#if __cccl_ptx_isa >= 200
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_ctaid_y()
{
  _CUDA_VSTD::uint32_t __sreg_value;
  asm("mov.u32 %0, %%ctaid.y;" : "=r"(__sreg_value) : :);
  return __sreg_value;
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u32 sreg_value, %%ctaid.z; // PTX ISA 20
template <typename = void>
__device__ static inline uint32_t get_sreg_ctaid_z();
*/
#if __cccl_ptx_isa >= 200
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_ctaid_z()
{
  _CUDA_VSTD::uint32_t __sreg_value;
  asm("mov.u32 %0, %%ctaid.z;" : "=r"(__sreg_value) : :);
  return __sreg_value;
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u32 sreg_value, %%nctaid.x; // PTX ISA 20
template <typename = void>
__device__ static inline uint32_t get_sreg_nctaid_x();
*/
#if __cccl_ptx_isa >= 200
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_nctaid_x()
{
  _CUDA_VSTD::uint32_t __sreg_value;
  asm("mov.u32 %0, %%nctaid.x;" : "=r"(__sreg_value) : :);
  return __sreg_value;
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u32 sreg_value, %%nctaid.y; // PTX ISA 20
template <typename = void>
__device__ static inline uint32_t get_sreg_nctaid_y();
*/
#if __cccl_ptx_isa >= 200
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_nctaid_y()
{
  _CUDA_VSTD::uint32_t __sreg_value;
  asm("mov.u32 %0, %%nctaid.y;" : "=r"(__sreg_value) : :);
  return __sreg_value;
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u32 sreg_value, %%nctaid.z; // PTX ISA 20
template <typename = void>
__device__ static inline uint32_t get_sreg_nctaid_z();
*/
#if __cccl_ptx_isa >= 200
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_nctaid_z()
{
  _CUDA_VSTD::uint32_t __sreg_value;
  asm("mov.u32 %0, %%nctaid.z;" : "=r"(__sreg_value) : :);
  return __sreg_value;
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u32 sreg_value, %%smid; // PTX ISA 13
template <typename = void>
__device__ static inline uint32_t get_sreg_smid();
*/
#if __cccl_ptx_isa >= 130
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_smid()
{
  _CUDA_VSTD::uint32_t __sreg_value;
  asm("mov.u32 %0, %%smid;" : "=r"(__sreg_value) : :);
  return __sreg_value;
}
#endif // __cccl_ptx_isa >= 130

/*
// mov.u32 sreg_value, %%nsmid; // PTX ISA 20, SM_35
template <typename = void>
__device__ static inline uint32_t get_sreg_nsmid();
*/
#if __cccl_ptx_isa >= 200
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_nsmid_is_not_supported_before_SM_35__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_nsmid()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 350
  _CUDA_VSTD::uint32_t __sreg_value;
  asm volatile("mov.u32 %0, %%nsmid;" : "=r"(__sreg_value) : :);
  return __sreg_value;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_get_sreg_nsmid_is_not_supported_before_SM_35__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u64 sreg_value, %%gridid; // PTX ISA 30
template <typename = void>
__device__ static inline uint64_t get_sreg_gridid();
*/
#if __cccl_ptx_isa >= 300
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint64_t get_sreg_gridid()
{
  _CUDA_VSTD::uint64_t __sreg_value;
  asm("mov.u64 %0, %%gridid;" : "=l"(__sreg_value) : :);
  return __sreg_value;
}
#endif // __cccl_ptx_isa >= 300

/*
// mov.pred sreg_value, %%is_explicit_cluster; // PTX ISA 78, SM_90
template <typename = void>
__device__ static inline bool get_sreg_is_explicit_cluster();
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_is_explicit_cluster_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline bool get_sreg_is_explicit_cluster()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  _CUDA_VSTD::uint32_t __sreg_value;
  asm("{\n\t .reg .pred P_OUT; \n\t"
      "mov.pred P_OUT, %%is_explicit_cluster;\n\t"
      "selp.b32 %0, 1, 0, P_OUT; \n"
      "}"
      : "=r"(__sreg_value)
      :
      :);
  return static_cast<bool>(__sreg_value);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_get_sreg_is_explicit_cluster_is_not_supported_before_SM_90__();
  return false;
#  endif
}
#endif // __cccl_ptx_isa >= 780

/*
// mov.u32 sreg_value, %%clusterid.x; // PTX ISA 78, SM_90
template <typename = void>
__device__ static inline uint32_t get_sreg_clusterid_x();
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_clusterid_x_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_clusterid_x()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  _CUDA_VSTD::uint32_t __sreg_value;
  asm("mov.u32 %0, %%clusterid.x;" : "=r"(__sreg_value) : :);
  return __sreg_value;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_get_sreg_clusterid_x_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 780

/*
// mov.u32 sreg_value, %%clusterid.y; // PTX ISA 78, SM_90
template <typename = void>
__device__ static inline uint32_t get_sreg_clusterid_y();
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_clusterid_y_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_clusterid_y()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  _CUDA_VSTD::uint32_t __sreg_value;
  asm("mov.u32 %0, %%clusterid.y;" : "=r"(__sreg_value) : :);
  return __sreg_value;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_get_sreg_clusterid_y_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 780

/*
// mov.u32 sreg_value, %%clusterid.z; // PTX ISA 78, SM_90
template <typename = void>
__device__ static inline uint32_t get_sreg_clusterid_z();
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_clusterid_z_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_clusterid_z()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  _CUDA_VSTD::uint32_t __sreg_value;
  asm("mov.u32 %0, %%clusterid.z;" : "=r"(__sreg_value) : :);
  return __sreg_value;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_get_sreg_clusterid_z_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 780

/*
// mov.u32 sreg_value, %%nclusterid.x; // PTX ISA 78, SM_90
template <typename = void>
__device__ static inline uint32_t get_sreg_nclusterid_x();
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_nclusterid_x_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_nclusterid_x()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  _CUDA_VSTD::uint32_t __sreg_value;
  asm("mov.u32 %0, %%nclusterid.x;" : "=r"(__sreg_value) : :);
  return __sreg_value;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_get_sreg_nclusterid_x_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 780

/*
// mov.u32 sreg_value, %%nclusterid.y; // PTX ISA 78, SM_90
template <typename = void>
__device__ static inline uint32_t get_sreg_nclusterid_y();
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_nclusterid_y_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_nclusterid_y()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  _CUDA_VSTD::uint32_t __sreg_value;
  asm("mov.u32 %0, %%nclusterid.y;" : "=r"(__sreg_value) : :);
  return __sreg_value;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_get_sreg_nclusterid_y_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 780

/*
// mov.u32 sreg_value, %%nclusterid.z; // PTX ISA 78, SM_90
template <typename = void>
__device__ static inline uint32_t get_sreg_nclusterid_z();
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_nclusterid_z_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_nclusterid_z()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  _CUDA_VSTD::uint32_t __sreg_value;
  asm("mov.u32 %0, %%nclusterid.z;" : "=r"(__sreg_value) : :);
  return __sreg_value;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_get_sreg_nclusterid_z_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 780

/*
// mov.u32 sreg_value, %%cluster_ctaid.x; // PTX ISA 78, SM_90
template <typename = void>
__device__ static inline uint32_t get_sreg_cluster_ctaid_x();
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_cluster_ctaid_x_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_cluster_ctaid_x()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  _CUDA_VSTD::uint32_t __sreg_value;
  asm("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(__sreg_value) : :);
  return __sreg_value;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_get_sreg_cluster_ctaid_x_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 780

/*
// mov.u32 sreg_value, %%cluster_ctaid.y; // PTX ISA 78, SM_90
template <typename = void>
__device__ static inline uint32_t get_sreg_cluster_ctaid_y();
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_cluster_ctaid_y_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_cluster_ctaid_y()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  _CUDA_VSTD::uint32_t __sreg_value;
  asm("mov.u32 %0, %%cluster_ctaid.y;" : "=r"(__sreg_value) : :);
  return __sreg_value;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_get_sreg_cluster_ctaid_y_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 780

/*
// mov.u32 sreg_value, %%cluster_ctaid.z; // PTX ISA 78, SM_90
template <typename = void>
__device__ static inline uint32_t get_sreg_cluster_ctaid_z();
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_cluster_ctaid_z_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_cluster_ctaid_z()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  _CUDA_VSTD::uint32_t __sreg_value;
  asm("mov.u32 %0, %%cluster_ctaid.z;" : "=r"(__sreg_value) : :);
  return __sreg_value;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_get_sreg_cluster_ctaid_z_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 780

/*
// mov.u32 sreg_value, %%cluster_nctaid.x; // PTX ISA 78, SM_90
template <typename = void>
__device__ static inline uint32_t get_sreg_cluster_nctaid_x();
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_cluster_nctaid_x_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_cluster_nctaid_x()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  _CUDA_VSTD::uint32_t __sreg_value;
  asm("mov.u32 %0, %%cluster_nctaid.x;" : "=r"(__sreg_value) : :);
  return __sreg_value;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_get_sreg_cluster_nctaid_x_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 780

/*
// mov.u32 sreg_value, %%cluster_nctaid.y; // PTX ISA 78, SM_90
template <typename = void>
__device__ static inline uint32_t get_sreg_cluster_nctaid_y();
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_cluster_nctaid_y_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_cluster_nctaid_y()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  _CUDA_VSTD::uint32_t __sreg_value;
  asm("mov.u32 %0, %%cluster_nctaid.y;" : "=r"(__sreg_value) : :);
  return __sreg_value;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_get_sreg_cluster_nctaid_y_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 780

/*
// mov.u32 sreg_value, %%cluster_nctaid.z; // PTX ISA 78, SM_90
template <typename = void>
__device__ static inline uint32_t get_sreg_cluster_nctaid_z();
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_cluster_nctaid_z_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_cluster_nctaid_z()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  _CUDA_VSTD::uint32_t __sreg_value;
  asm("mov.u32 %0, %%cluster_nctaid.z;" : "=r"(__sreg_value) : :);
  return __sreg_value;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_get_sreg_cluster_nctaid_z_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 780

/*
// mov.u32 sreg_value, %%cluster_ctarank; // PTX ISA 78, SM_90
template <typename = void>
__device__ static inline uint32_t get_sreg_cluster_ctarank();
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_cluster_ctarank_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_cluster_ctarank()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  _CUDA_VSTD::uint32_t __sreg_value;
  asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(__sreg_value) : :);
  return __sreg_value;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_get_sreg_cluster_ctarank_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 780

/*
// mov.u32 sreg_value, %%cluster_nctarank; // PTX ISA 78, SM_90
template <typename = void>
__device__ static inline uint32_t get_sreg_cluster_nctarank();
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_cluster_nctarank_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_cluster_nctarank()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  _CUDA_VSTD::uint32_t __sreg_value;
  asm("mov.u32 %0, %%cluster_nctarank;" : "=r"(__sreg_value) : :);
  return __sreg_value;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_get_sreg_cluster_nctarank_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 780

/*
// mov.u32 sreg_value, %%lanemask_eq; // PTX ISA 20, SM_35
template <typename = void>
__device__ static inline uint32_t get_sreg_lanemask_eq();
*/
#if __cccl_ptx_isa >= 200
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_lanemask_eq_is_not_supported_before_SM_35__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_lanemask_eq()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 350
  _CUDA_VSTD::uint32_t __sreg_value;
  asm("mov.u32 %0, %%lanemask_eq;" : "=r"(__sreg_value) : :);
  return __sreg_value;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_get_sreg_lanemask_eq_is_not_supported_before_SM_35__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u32 sreg_value, %%lanemask_le; // PTX ISA 20, SM_35
template <typename = void>
__device__ static inline uint32_t get_sreg_lanemask_le();
*/
#if __cccl_ptx_isa >= 200
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_lanemask_le_is_not_supported_before_SM_35__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_lanemask_le()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 350
  _CUDA_VSTD::uint32_t __sreg_value;
  asm("mov.u32 %0, %%lanemask_le;" : "=r"(__sreg_value) : :);
  return __sreg_value;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_get_sreg_lanemask_le_is_not_supported_before_SM_35__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u32 sreg_value, %%lanemask_lt; // PTX ISA 20, SM_35
template <typename = void>
__device__ static inline uint32_t get_sreg_lanemask_lt();
*/
#if __cccl_ptx_isa >= 200
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_lanemask_lt_is_not_supported_before_SM_35__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_lanemask_lt()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 350
  _CUDA_VSTD::uint32_t __sreg_value;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(__sreg_value) : :);
  return __sreg_value;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_get_sreg_lanemask_lt_is_not_supported_before_SM_35__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u32 sreg_value, %%lanemask_ge; // PTX ISA 20, SM_35
template <typename = void>
__device__ static inline uint32_t get_sreg_lanemask_ge();
*/
#if __cccl_ptx_isa >= 200
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_lanemask_ge_is_not_supported_before_SM_35__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_lanemask_ge()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 350
  _CUDA_VSTD::uint32_t __sreg_value;
  asm("mov.u32 %0, %%lanemask_ge;" : "=r"(__sreg_value) : :);
  return __sreg_value;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_get_sreg_lanemask_ge_is_not_supported_before_SM_35__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u32 sreg_value, %%lanemask_gt; // PTX ISA 20, SM_35
template <typename = void>
__device__ static inline uint32_t get_sreg_lanemask_gt();
*/
#if __cccl_ptx_isa >= 200
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_lanemask_gt_is_not_supported_before_SM_35__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_lanemask_gt()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 350
  _CUDA_VSTD::uint32_t __sreg_value;
  asm("mov.u32 %0, %%lanemask_gt;" : "=r"(__sreg_value) : :);
  return __sreg_value;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_get_sreg_lanemask_gt_is_not_supported_before_SM_35__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u32 sreg_value, %%clock; // PTX ISA 10
template <typename = void>
__device__ static inline uint32_t get_sreg_clock();
*/
#if __cccl_ptx_isa >= 100
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_clock()
{
  _CUDA_VSTD::uint32_t __sreg_value;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(__sreg_value) : :);
  return __sreg_value;
}
#endif // __cccl_ptx_isa >= 100

/*
// mov.u32 sreg_value, %%clock_hi; // PTX ISA 50, SM_35
template <typename = void>
__device__ static inline uint32_t get_sreg_clock_hi();
*/
#if __cccl_ptx_isa >= 500
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_clock_hi_is_not_supported_before_SM_35__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_clock_hi()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 350
  _CUDA_VSTD::uint32_t __sreg_value;
  asm volatile("mov.u32 %0, %%clock_hi;" : "=r"(__sreg_value) : :);
  return __sreg_value;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_get_sreg_clock_hi_is_not_supported_before_SM_35__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 500

/*
// mov.u64 sreg_value, %%clock64; // PTX ISA 20, SM_35
template <typename = void>
__device__ static inline uint64_t get_sreg_clock64();
*/
#if __cccl_ptx_isa >= 200
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_clock64_is_not_supported_before_SM_35__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint64_t get_sreg_clock64()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 350
  _CUDA_VSTD::uint64_t __sreg_value;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(__sreg_value) : :);
  return __sreg_value;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_get_sreg_clock64_is_not_supported_before_SM_35__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u64 sreg_value, %%globaltimer; // PTX ISA 31, SM_35
template <typename = void>
__device__ static inline uint64_t get_sreg_globaltimer();
*/
#if __cccl_ptx_isa >= 310
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_globaltimer_is_not_supported_before_SM_35__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint64_t get_sreg_globaltimer()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 350
  _CUDA_VSTD::uint64_t __sreg_value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(__sreg_value) : :);
  return __sreg_value;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_get_sreg_globaltimer_is_not_supported_before_SM_35__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 310

/*
// mov.u32 sreg_value, %%globaltimer_lo; // PTX ISA 31, SM_35
template <typename = void>
__device__ static inline uint32_t get_sreg_globaltimer_lo();
*/
#if __cccl_ptx_isa >= 310
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_globaltimer_lo_is_not_supported_before_SM_35__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_globaltimer_lo()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 350
  _CUDA_VSTD::uint32_t __sreg_value;
  asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(__sreg_value) : :);
  return __sreg_value;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_get_sreg_globaltimer_lo_is_not_supported_before_SM_35__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 310

/*
// mov.u32 sreg_value, %%globaltimer_hi; // PTX ISA 31, SM_35
template <typename = void>
__device__ static inline uint32_t get_sreg_globaltimer_hi();
*/
#if __cccl_ptx_isa >= 310
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_globaltimer_hi_is_not_supported_before_SM_35__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_globaltimer_hi()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 350
  _CUDA_VSTD::uint32_t __sreg_value;
  asm volatile("mov.u32 %0, %%globaltimer_hi;" : "=r"(__sreg_value) : :);
  return __sreg_value;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_get_sreg_globaltimer_hi_is_not_supported_before_SM_35__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 310

/*
// mov.u32 sreg_value, %%total_smem_size; // PTX ISA 41, SM_35
template <typename = void>
__device__ static inline uint32_t get_sreg_total_smem_size();
*/
#if __cccl_ptx_isa >= 410
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_total_smem_size_is_not_supported_before_SM_35__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_total_smem_size()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 350
  _CUDA_VSTD::uint32_t __sreg_value;
  asm("mov.u32 %0, %%total_smem_size;" : "=r"(__sreg_value) : :);
  return __sreg_value;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_get_sreg_total_smem_size_is_not_supported_before_SM_35__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 410

/*
// mov.u32 sreg_value, %%aggr_smem_size; // PTX ISA 81, SM_90
template <typename = void>
__device__ static inline uint32_t get_sreg_aggr_smem_size();
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_aggr_smem_size_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_aggr_smem_size()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  _CUDA_VSTD::uint32_t __sreg_value;
  asm("mov.u32 %0, %%aggr_smem_size;" : "=r"(__sreg_value) : :);
  return __sreg_value;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_get_sreg_aggr_smem_size_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// mov.u32 sreg_value, %%dynamic_smem_size; // PTX ISA 41, SM_35
template <typename = void>
__device__ static inline uint32_t get_sreg_dynamic_smem_size();
*/
#if __cccl_ptx_isa >= 410
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_dynamic_smem_size_is_not_supported_before_SM_35__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_dynamic_smem_size()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 350
  _CUDA_VSTD::uint32_t __sreg_value;
  asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(__sreg_value) : :);
  return __sreg_value;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_get_sreg_dynamic_smem_size_is_not_supported_before_SM_35__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 410

/*
// mov.u64 sreg_value, %%current_graph_exec; // PTX ISA 80, SM_50
template <typename = void>
__device__ static inline uint64_t get_sreg_current_graph_exec();
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_current_graph_exec_is_not_supported_before_SM_50__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint64_t get_sreg_current_graph_exec()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 500
  _CUDA_VSTD::uint64_t __sreg_value;
  asm("mov.u64 %0, %%current_graph_exec;" : "=l"(__sreg_value) : :);
  return __sreg_value;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_get_sreg_current_graph_exec_is_not_supported_before_SM_50__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 800

#endif // _CUDA_PTX_GENERATED_GET_SREG_H_
