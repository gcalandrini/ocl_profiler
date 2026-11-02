/*
 * ocl_wrapper.c
 *
 *  Created on: Oct 30, 2013
 *      Author: Guilherme Calandrini
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <errno.h>

#define __USE_GNU
#include <dlfcn.h>

#ifdef __APPLE__
#include <OpenCL/OpenCL.h>
#else
#include <CL/cl.h>
#endif

#include "nvprof.h"

#define OUTPUT_SIZE (0x10000)
#define PROFILE_ERR "Failed to generate profile output"

char kname[255];
cl_uint contextid = 0;
cl_char platform_info[1000];
cl_uint deviceid = 0;
cl_char devname[255];
char *fileName;

int getdevprop(cl_device_id devices) {

	clGetDeviceInfo(devices, CL_DEVICE_NAME, sizeof(devname), &devname, NULL);

	return 0;
}

int
read_tsc(uint64_t *tsc)
{
    uint32_t d;
    uint32_t a;

    asm("rdtsc;"
        "mov %%edx, %0;"      // move edx into d
        "mov %%eax, %1;"      // move eax into a
        : "=r" (d), "=r" (a)  // output
        :                     // input
        : "%edx", "eax"       // clobbered regiters
        );

    *tsc = (uint64_t)d << 32 | a;

    return 0;
}

void printoutput(cl_ulong time_start, cl_ulong time_end,
		  struct timespec start, struct timespec end, char* method) {

	    printf("######################################\n");
	    printf("OCL method: %s\n", method);
		printf("OCL time_start: %lu nsec\n", time_start);
		printf("OCL time_end: %lu nsec\n", time_end);
		printf("OCL elapsed: %lu nsec\n", time_end - time_start);
		uint64_t tsc;
		read_tsc(&tsc);
		printf("%lu\n", tsc);

	    double adjusfact = (double)(((start.tv_sec*1000000000) + start.tv_nsec)/(long)time_start);
//double adjusfact = 1;
	    printf("--------------------------------------\n");
	    printf("adjust factor %f\n", (double)((long)time_start)/((start.tv_sec*1000000000) + start.tv_nsec));
		printf("CPU Start Cpu: %ld nsec\n", ((start.tv_sec*1000000000) + start.tv_nsec)/(long)adjusfact);
		printf("CPU End Cpu: %ld nsec\n", ((end.tv_sec*1000000000) + end.tv_nsec)/(long)adjusfact);
		//printf("CPU End Cpu: %ld milliseconds\n", ((end.tv_sec*1000000000) + end.tv_nsec)/169000);

		long elapsed_cputime = ((end.tv_sec*1000000000) + end.tv_nsec) - ((start.tv_sec*1000000000) + start.tv_nsec);
		printf("CPU Elapsed Cpu: %ld milliseconds\n", elapsed_cputime);

		printf("  #kernel: %s, time: %f sec\n", kname, (double)((time_end - time_start)/1e9));
		printf("######################################\n\n");
}


/* Getting platforms available */
int clGetPlatformIDs(cl_uint entries, cl_platform_id *platform_id, cl_uint *nplatforms) {

	static int (*clGetPlatform_real)(cl_uint, cl_platform_id*, cl_uint*)=NULL;

	#ifdef DEBUG
		printf("#* -- > Wrapping clGetPlatformIDs\n");
	#endif

	if (!clGetPlatform_real) clGetPlatform_real=dlsym(RTLD_NEXT,"clGetPlatformIDs");

	return clGetPlatform_real(entries, platform_id, nplatforms);
}

/* Getting devices available */
extern CL_API_ENTRY cl_int CL_API_CALL
clGetDeviceIDs(cl_platform_id platform_id, cl_device_type DEVTYPE, cl_uint entries,
               cl_device_id *devices, cl_uint *ndevices) {

	static cl_int (*clGetDeviceIDs_real)(cl_platform_id platform_id, cl_device_type DEVTYPE, cl_uint entries,
                                         cl_device_id *devices, cl_uint *ndevices)=NULL;

	#ifdef DEBUG
		printf("#* -- > Wrapping clGetDeviceIDs\n");
    #endif

	if (!clGetDeviceIDs_real) clGetDeviceIDs_real=dlsym(RTLD_NEXT,"clGetDeviceIDs");

	return clGetDeviceIDs_real(platform_id, DEVTYPE, entries, devices, ndevices);
}

/* Create kernel */
cl_kernel clCreateKernel(cl_program program, const char *kernel_name, cl_int *err) {

	static cl_kernel (*clCreateKernel_real)(cl_program, const char*, cl_int*)=NULL;
	strcpy(kname, kernel_name);

	#ifdef DEBUG
	 	printf("#* -- > Wrapping clCreateKernel\n");
		printf("#* Kernel name: %s\n", kname);
	#endif

	if (!clCreateKernel_real) clCreateKernel_real=dlsym(RTLD_NEXT,"clCreateKernel");

	return clCreateKernel_real(program, kernel_name, err);
}

/* Creating command queue */
cl_command_queue clCreateCommandQueue(cl_context context, cl_device_id device_id,
		                              cl_command_queue_properties CMDQUEUEPROP, cl_int *err){

	static cl_command_queue (*clCreateCommandQueue_real)(cl_context context, cl_device_id device_id,
			                                cl_command_queue_properties CMDQUEUEPROP, cl_int *err)=NULL;

	if (!clCreateCommandQueue_real) clCreateCommandQueue_real=dlsym(RTLD_NEXT,"clCreateCommandQueue");
	//printf("deviceid: %d", &device_id);
	getdevprop(device_id);

	#ifdef DEBUG
		printf("#* -- > Wrapping clCreateCommandQueue\n");
		printf("#* -- > Enabling profile\n");
	#endif

	if ( profout(0, 0) != 0 )
		printf(PROFILE_ERR);

	return clCreateCommandQueue_real(context, device_id, CL_QUEUE_PROFILING_ENABLE, err);
}

int clGetEventProfilingInfo(cl_event event, cl_profiling_info PROFINFO, size_t size,
                            void *value, size_t *size_ret) {

	static int (*clGetEventProfilingInfo_real)(cl_event, cl_profiling_info, size_t,
                                               void *, size_t *)=NULL;

	#ifdef DEBUG
		printf("#* -- > Wrapping clGetEventProfilingInfo\n");
	#endif

	if (!clGetEventProfilingInfo_real) clGetEventProfilingInfo_real=dlsym(RTLD_NEXT,"clGetEventProfilingInfo");

	return clGetEventProfilingInfo_real(event, PROFINFO, size, value, size_ret);
}


extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer,
                     cl_bool BLOCK, size_t offset, size_t size, const void *ete,
                     cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {

	static int (*clEnqueueWriteBuffer_real)(cl_command_queue command_queue, cl_mem buffer,
            cl_bool BLOCK, size_t offset, size_t size, const void *ete,
            cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event)=NULL;

	#ifdef DEBUG
		printf("#* -- > Wrapping clEnqueueWriteBuffer\n");
	#endif

	if (!clEnqueueWriteBuffer_real) clEnqueueWriteBuffer_real=dlsym(RTLD_NEXT,"clEnqueueWriteBuffer");

	/* Incubating */
	cl_ulong dvtime_start, dvtime_end;
	struct timespec hstart, hend;
	cl_event *wrp_event = malloc(sizeof(cl_event));

	if (event == NULL) {
			clock_gettime(CLOCK_REALTIME, &hstart);
			clEnqueueWriteBuffer_real(command_queue, buffer, BLOCK, offset,
									  size, ete, num_events_in_wait_list,
									  event_wait_list, wrp_event);

		    clFinish(command_queue);
		    clock_gettime(CLOCK_REALTIME, &hend);

		    clGetEventProfilingInfo(*wrp_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &dvtime_start, NULL);
		    clGetEventProfilingInfo(*wrp_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &dvtime_end, NULL);
		} else {
			clock_gettime(CLOCK_REALTIME, &hstart);
			clEnqueueWriteBuffer_real(command_queue, buffer, BLOCK, offset,
									  size, ete, num_events_in_wait_list,
									  event_wait_list, event);

		    clFinish(command_queue);
		    clock_gettime(CLOCK_REALTIME, &hend);
			clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &dvtime_start, NULL);
			clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &dvtime_end, NULL);
		}

	if ( nvprofout(dvtime_start, dvtime_end, hstart, hend, "memcpyHtoD") != 0 ) {
		printf(PROFILE_ERR);
	}

	return clEnqueueWriteBuffer_real(command_queue, buffer, BLOCK, offset, size,
			ete, num_events_in_wait_list, event_wait_list, event);

}

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer,
                     cl_bool BLOCK, size_t offset, size_t size, void *ete,
                     cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {

	static int (*clEnqueueReadBuffer_real)(cl_command_queue command_queue, cl_mem buffer,
            cl_bool BLOCK, size_t offset, size_t size, void *ete,
            cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event)=NULL;

	#ifdef DEBUG
		printf("#* -- > Wrapping clEnqueueReadBuffer\n");
	#endif

	if (!clEnqueueReadBuffer_real) clEnqueueReadBuffer_real=dlsym(RTLD_NEXT,"clEnqueueReadBuffer");

	/* Incubating */
	cl_ulong dvtime_start, dvtime_end;
	struct timespec hstart, hend;
	cl_event *wrp_event = malloc(sizeof(cl_event));

	if (event == NULL) {
			clock_gettime(CLOCK_REALTIME, &hstart);
			clEnqueueReadBuffer_real(command_queue, buffer, BLOCK, offset,
									  size, ete, num_events_in_wait_list,
									  event_wait_list, wrp_event);

		    clFinish(command_queue);
		    clock_gettime(CLOCK_REALTIME, &hend);

		    clGetEventProfilingInfo(*wrp_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &dvtime_start, NULL);
		    clGetEventProfilingInfo(*wrp_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &dvtime_end, NULL);
		} else {
			clock_gettime(CLOCK_REALTIME, &hstart);
			clEnqueueReadBuffer_real(command_queue, buffer, BLOCK, offset,
									  size, ete, num_events_in_wait_list,
									  event_wait_list, event);

		    clFinish(command_queue);
		    clock_gettime(CLOCK_REALTIME, &hend);
			clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &dvtime_start, NULL);
			clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &dvtime_end, NULL);
		}

	if ( nvprofout(dvtime_start, dvtime_end, hstart, hend, "memcpyDtoH") != 0 ) {
		printf(PROFILE_ERR);
	}

	return clEnqueueReadBuffer_real(command_queue, buffer, BLOCK, offset, size,
			ete, num_events_in_wait_list, event_wait_list, event);

}

/* Getting platforms available */
extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel,
						   cl_uint work_dim, const size_t *global_work_offset,
						   const size_t *global_work_size, const size_t *local_work_size,
						   cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {

	#ifdef DEBUG
		printf("\n#* -- > Wrapping clEnqueueNDRangeKernel...\n");
	#endif

	static int (*clEnqueueNDRangeKernel_real)(cl_command_queue command_queue, cl_kernel kernel,
			   cl_uint work_dim, const size_t *global_work_offset,
			   const size_t *global_work_size, const size_t *local_work_size,
			   cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) = NULL;

	/* Incubating */
	cl_ulong dvtime_start, dvtime_end;
	struct timespec hstart, hend;
	cl_event *wrp_event = malloc(sizeof(cl_event));

	//printf("work-dim: %d\n", work_dim);
	//printf("global size: %zu\n", global_work_size[0]);
	//printf("local size: %zu\n", local_work_size[0]); // TODO: check if null

	if (!clEnqueueNDRangeKernel_real)
		clEnqueueNDRangeKernel_real = dlsym(RTLD_NEXT,"clEnqueueNDRangeKernel");

	if (event == NULL) {
		clock_gettime(CLOCK_REALTIME, &hstart);
		clEnqueueNDRangeKernel_real(command_queue, kernel, work_dim, global_work_offset,
									global_work_size, local_work_size, num_events_in_wait_list,
									event_wait_list, wrp_event);

	    clFinish(command_queue);
	    clock_gettime(CLOCK_REALTIME, &hend);

	    clGetEventProfilingInfo(*wrp_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &dvtime_start, NULL);
	    clGetEventProfilingInfo(*wrp_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &dvtime_end, NULL);
	} else {
		clock_gettime(CLOCK_REALTIME, &hstart);
		clEnqueueNDRangeKernel_real(command_queue, kernel, work_dim, global_work_offset,
									global_work_size, local_work_size, num_events_in_wait_list,
									event_wait_list, event);

	    clFinish(command_queue);
	    clock_gettime(CLOCK_REALTIME, &hend);
		clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &dvtime_start, NULL);
		clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &dvtime_end, NULL);
	}

	#ifdef DEBUG
		printoutput(dvtime_start, dvtime_end, hstart, hend, kname);
    #endif

	if ( nvprofout(dvtime_start, dvtime_end, hstart, hend, kname) != 0 ) {
		printf(PROFILE_ERR);
	}

	return 0;
}

