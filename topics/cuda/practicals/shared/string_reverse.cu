#include <cstdlib>
#include <cstdio>
#include <iostream>

#include "util.hpp"

// TODO : implement a kernel that reverses a string of length n in place
__global__
void reverse_string(char* str, int n) {
	extern char __shared__ str_out[];
	auto i = threadIdx.x;
	str_out[i] = str[n - i - 1];
	__syncthreads();
	str[i] = str_out[i];
}




int main(int argc, char** argv) {
    // check that the user has passed a string to reverse
    if(argc<2) {
        std::cout << "useage : ./string_reverse \"string to reverse\"\n" << std::endl;
        exit(0);
    }

    // determine the length of the string, and copy in to buffer
    auto n = strlen(argv[1]);
    auto string = malloc_managed<char>(n+1);
    std::copy(argv[1], argv[1]+n, string);
    string[n] = 0; // add null terminator

    std::cout << "string to reverse:\n" << string << "\n";

    // TODO : call the string reverse function
    reverse_string<<<1, n, n>>>(string, n);
    // print reversed string
    cudaDeviceSynchronize();
    std::cout << "reversed string:\n" << string << "\n";

    // free memory
    cudaFree(string);

    return 0;
}

