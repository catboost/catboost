/*******************************************************************************
* Copyright 1999-2022 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
!  Content:
!      Intel(R) oneAPI Math Kernel Library (oneMKL) interface for service routines
!******************************************************************************/

#ifndef _MKL_SERVICE_H_
#define _MKL_SERVICE_H_

#include <stdlib.h>
#include "mkl_types.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

void    MKL_Get_Version(MKLVersion *ver); /* Returns information about the version of the oneMKL software */
#define mkl_get_version             MKL_Get_Version

void    MKL_Get_Version_String(char *buffer, int len); /* Returns a string that contains oneMKL library version information */
#define mkl_get_version_string      MKL_Get_Version_String

void    MKL_Free_Buffers(void); /* Frees the memory allocated by the oneMKL Memory Allocator */
#define mkl_free_buffers            MKL_Free_Buffers

void    MKL_Thread_Free_Buffers(void); /* Frees the memory allocated by the oneMKL Memory Allocator in the current thread only */
#define mkl_thread_free_buffers     MKL_Thread_Free_Buffers

MKL_INT64 MKL_Mem_Stat(int* nbuffers); /* oneMKL Memory Allocator statistical information. */
                                       /* Returns an amount of memory, allocated by the oneMKL Memory Allocator */
                                       /* in <nbuffers> buffers. */
#define mkl_mem_stat                MKL_Mem_Stat

#define  MKL_PEAK_MEM_DISABLE       0
#define  MKL_PEAK_MEM_ENABLE        1
#define  MKL_PEAK_MEM_RESET        -1
#define  MKL_PEAK_MEM               2
MKL_INT64 MKL_Peak_Mem_Usage(int reset); /* Returns the peak amount of memory, allocated by the oneMKL Memory Allocator */
#define mkl_peak_mem_usage          MKL_Peak_Mem_Usage

void*   MKL_malloc(size_t size, int align); /* Allocates the aligned buffer */
#define mkl_malloc                  MKL_malloc

void*   MKL_calloc(size_t num, size_t size, int align); /* Allocates the aligned num*size - bytes memory buffer initialized by zeros */
#define mkl_calloc                  MKL_calloc

void*   MKL_realloc(void *ptr, size_t size); /* Changes the size of memory buffer allocated by MKL_malloc/MKL_calloc */
#define mkl_realloc                 MKL_realloc

void    MKL_free(void *ptr); /* Frees the memory allocated by MKL_malloc() */
#define mkl_free                    MKL_free

int     MKL_Disable_Fast_MM(void); /* Turns off the oneMKL Memory Allocator */
#define  mkl_disable_fast_mm        MKL_Disable_Fast_MM

void    MKL_Get_Cpu_Clocks(unsigned MKL_INT64 *); /* Gets CPU clocks */
#define mkl_get_cpu_clocks          MKL_Get_Cpu_Clocks

double  MKL_Get_Cpu_Frequency(void); /* Gets CPU frequency in GHz */
#define mkl_get_cpu_frequency       MKL_Get_Cpu_Frequency

double  MKL_Get_Max_Cpu_Frequency(void); /* Gets max CPU frequency in GHz */
#define mkl_get_max_cpu_frequency   MKL_Get_Max_Cpu_Frequency

double  MKL_Get_Clocks_Frequency(void); /* Gets clocks frequency in GHz */
#define mkl_get_clocks_frequency    MKL_Get_Clocks_Frequency

int     MKL_Set_Num_Threads_Local(int nth);
#define mkl_set_num_threads_local   MKL_Set_Num_Threads_Local
void    MKL_Set_Num_Threads(int nth);
#define mkl_set_num_threads         MKL_Set_Num_Threads
int     MKL_Get_Max_Threads(void);
#define mkl_get_max_threads         MKL_Get_Max_Threads
void    MKL_Set_Num_Stripes(int nstripes);
#define mkl_set_num_stripes         MKL_Set_Num_Stripes
int     MKL_Get_Num_Stripes(void);
#define mkl_get_num_stripes         MKL_Get_Num_Stripes
int     MKL_Domain_Set_Num_Threads(int nth, int MKL_DOMAIN);
#define mkl_domain_set_num_threads  MKL_Domain_Set_Num_Threads
int     MKL_Domain_Get_Max_Threads(int MKL_DOMAIN);
#define mkl_domain_get_max_threads  MKL_Domain_Get_Max_Threads
void    MKL_Set_Dynamic(int bool_MKL_DYNAMIC);
#define mkl_set_dynamic             MKL_Set_Dynamic
int     MKL_Get_Dynamic(void);
#define mkl_get_dynamic             MKL_Get_Dynamic

/* oneMKL Progress routine */
#ifndef _MKL_PROGRESS_H_
#define _MKL_PROGRESS_H_
int     MKL_PROGRESS ( int* thread, int* step, char* stage, int lstage );
int     MKL_PROGRESS_( int* thread, int* step, char* stage, int lstage );
int     mkl_progress ( int* thread, int* step, char* stage, int lstage );
int     mkl_progress_( int* thread, int* step, char* stage, int lstage );
#endif /* _MKL_PROGRESS_H_ */

int     MKL_Enable_Instructions(int);
#define  mkl_enable_instructions    MKL_Enable_Instructions
#define  MKL_ENABLE_SSE4_2          0
#define  MKL_ENABLE_AVX             1
#define  MKL_ENABLE_AVX2            2
#define  MKL_ENABLE_AVX512_MIC      3
#define  MKL_ENABLE_AVX512          4
#define  MKL_ENABLE_AVX512_MIC_E1   5
#define  MKL_ENABLE_AVX512_E1       6
#define  MKL_ENABLE_AVX512_E2       7
#define  MKL_ENABLE_AVX512_E3       8
#define  MKL_ENABLE_AVX512_E4       9
#define  MKL_ENABLE_AVX2_E1        10
#define  MKL_SINGLE_PATH_ENABLE     0x0600

/* Single Dynamic library interface */
#define MKL_INTERFACE_LP64          0
#define MKL_INTERFACE_ILP64         1
#define MKL_INTERFACE_GNU           2
int     MKL_Set_Interface_Layer(int code);
#define mkl_set_interface_layer     MKL_Set_Interface_Layer

/* Single Dynamic library threading */
#define MKL_THREADING_INTEL         0
#define MKL_THREADING_SEQUENTIAL    1
#define MKL_THREADING_PGI           2
#define MKL_THREADING_GNU           3
#define MKL_THREADING_TBB           4
int     MKL_Set_Threading_Layer(int code);
#define mkl_set_threading_layer     MKL_Set_Threading_Layer

typedef void (* XerblaEntry) (const char * Name, const int * Num, const int Len);
XerblaEntry mkl_set_xerbla(XerblaEntry xerbla);

typedef int (* ProgressEntry) (int* thread, int* step, char* stage, int stage_len);
ProgressEntry mkl_set_progress(ProgressEntry progress);

typedef int (* PardisopivotEntry) (double* aii, double* bii, double*eps);
PardisopivotEntry mkl_set_pardiso_pivot(PardisopivotEntry pardiso_pivot);

/* oneMKL CBWR */
int     MKL_CBWR_Get(int);
#define mkl_cbwr_get                MKL_CBWR_Get
int     MKL_CBWR_Set(int);
#define mkl_cbwr_set                MKL_CBWR_Set
int     MKL_CBWR_Get_Auto_Branch(void);
#define mkl_cbwr_get_auto_branch    MKL_CBWR_Get_Auto_Branch

/* oneMKL Verbose */
int MKL_Set_Env_Mode(int);
#define mkl_set_env_mode            MKL_Set_Env_Mode

int MKL_Verbose(int);
#define mkl_verbose                 MKL_Verbose

int MKL_Verbose_Output_File(const char *fname);
#define mkl_verbose_output_file     MKL_Verbose_Output_File

#define MKL_EXIT_UNSUPPORTED_CPU    1
#define MKL_EXIT_CORRUPTED_INSTALL  2
#define MKL_EXIT_NO_MEMORY          3

#ifndef __MKLExitHandler
#define __MKLExitHandler
typedef void (* MKLExitHandler)(int why);
#endif
void MKL_Set_Exit_Handler(MKLExitHandler h);
#define mkl_set_exit_handler       MKL_Set_Exit_Handler

/* oneMKL MPI */
enum {
        MKL_BLACS_CUSTOM = 0,
        MKL_BLACS_MSMPI = 1,
        MKL_BLACS_INTELMPI = 2,
#if !defined(_WIN32) & !defined(_WIN64)
        MKL_BLACS_MPICH2 = 3,
        MKL_BLACS_LASTMPI = 4
#else
        MKL_BLACS_LASTMPI = 3
#endif
};
int MKL_Set_mpi(int vendor, const char *custom_library_name);
#define mkl_set_mpi MKL_Set_mpi

/* oneMKL Memory control */
#define MKL_MEM_MCDRAM 1

int MKL_Set_Memory_Limit(int mem_type, size_t limit);
#define mkl_set_memory_limit MKL_Set_Memory_Limit

void MKL_Finalize(void);
#define mkl_finalize MKL_Finalize

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _MKL_SERVICE_H_ */
