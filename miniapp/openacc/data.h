#pragma once

#include <iostream>
#include <cassert>
#include <openacc.h>

namespace data
{

// define some helper types that can be used to pass simulation
// data around without haveing to pass individual parameters
struct Discretization
{
    int nx;       // x dimension
    int ny;       // y dimension
    int N;        // grid dimension (nx*ny)
    int nt;       // number of time steps
    double dt;    // time step size
    double dx;    // distance between grid points
    double alpha; // dx^2/(D*dt)
};

// thin wrapper around a pointer that can be accessed as either a 2D or 1D array
// Field has dimension xdim * ydim in 2D, or length=xdim*ydim in 1D
class Field {
    public:
    // default constructor
    Field()
    :   xdim_(0),
        ydim_(0),
        ptr_(nullptr)
    {
		#pragma acc enter data copyin(this)
    };

    // constructor
    Field(int xdim, int ydim)
    :   xdim_(xdim),
        ydim_(ydim),
        ptr_(nullptr)
    {
		#pragma acc enter data copyin(this)
        init(xdim, ydim);
    };

    // destructor
    ~Field() {
        free();
		#pragma acc exit data delete(this)
    }

    void init(int xdim, int ydim) {
        #ifdef DEBUG
        assert(xdim>0 && ydim>0);
        #endif

        free();
        allocate(xdim, ydim);
        fill(0.);
    }

    double*       host_data()         { return ptr_; }
    const double* host_data()   const { return ptr_; }

    double*       device_data()       { return (double *) acc_deviceptr(ptr_); }
    const double* device_data() const { return (double *) acc_deviceptr(ptr_); }

    // access via (i,j) pair
	#pragma acc routine
    inline double&       operator() (int i, int j)        {
        #ifdef DEBUG
        assert(i>=0 && i<xdim_ && j>=0 && j<ydim_);
        #endif
        return ptr_[i+j*xdim_];
    }

	#pragma acc routine
    inline double const& operator() (int i, int j) const  {
        #ifdef DEBUG
        assert(i>=0 && i<xdim_ && j>=0 && j<ydim_);
        #endif
        return ptr_[i+j*xdim_];
    }

    // access as a 1D field
	#pragma acc routine
    inline double      & operator[] (int i) {
        #ifdef DEBUG
        assert(i>=0 && i<xdim_*ydim_);
        #endif
        return ptr_[i];
    }

	#pragma acc routine
    inline double const& operator[] (int i) const {
        #ifdef DEBUG
        assert(i>=0 && i<xdim_*ydim_);
        #endif
        return ptr_[i];
    }

	#pragma acc routine
    int xdim()   const { return xdim_; }
	#pragma acc routine
    int ydim()   const { return ydim_; }
	#pragma acc routine
    int length() const { return xdim_*ydim_; }

    /////////////////////////////////////////////////
    // helpers for coordinating host-device transfers
    /////////////////////////////////////////////////
    void update_host() {
        // xTODO: Update the host copy of the data
		#pragma acc update host(ptr_[:xdim_*ydim_])
    }

    void update_device() {
        // xTODO: Update the device copy of the data
		#pragma acc update device(ptr_[:xdim_*ydim_])
    }

    private:

    void allocate(int xdim, int ydim) {
        xdim_ = xdim;
        ydim_ = ydim;
        ptr_ = new double[xdim*ydim];
		#pragma acc update device(this)
		#pragma acc enter data copyin(ptr_[:xdim*ydim])
		// xTODO: Copy the whole object to the GPU.
        //       Pay attention to the order of the copies so that the data
        //       pointed to by `ptr_` is properly attached to the GPU's copy of
        //       this object.
    }

    // set to a constant value
    void fill(double val) {
        // initialize the host and device copy at the same time
        // xTODO: Offload this loop to the GPU
		#pragma acc parallel loop present(ptr_)
        for(int i=0; i<xdim_*ydim_; ++i)
            ptr_[i] = val;

        #pragma omp parallel for
        for(int i=0; i<xdim_*ydim_; ++i)
            ptr_[i] = val;
    }

    void free() {
        if (ptr_) {
            // xTODO: Delete the copy of this object from the GPU
			#pragma acc exit data delete(ptr_[:xdim_*ydim_])
            // NOTE: You will see some OpenACC runtime errors when your program exits
            //       This is a problem with the PGI runtime; you may ignore them.
            delete[] ptr_;
        }

        ptr_ = nullptr;
    }

    double* ptr_;
    int xdim_;
    int ydim_;
};

// fields that hold the solution
extern Field x_new; // 2d
extern Field x_old; // 2d

// fields that hold the boundary values
extern Field bndN; // 1d
extern Field bndE; // 1d
extern Field bndS; // 1d
extern Field bndW; // 1d

extern Discretization options;

} // namespace data
