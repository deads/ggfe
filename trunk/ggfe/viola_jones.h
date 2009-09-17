#ifndef VIOLA_JONES_H
#define VIOLA_JONES_H

#include <cvd/image.h>
#include <vector>
#include <utility>
#include <iostream>
#include <cvd/integral_image.h>

//This struct contains four points in a rectangle. They are stored as offsets
//relative to a pointer. Therefore the width of the image must be known to 
//construct one of these.
struct Quad
{
	int topleft, topright, bottomleft, bottomright;
};

typedef std::pair<CVD::ImageRef, CVD::ImageRef> Rectangle;
typedef std::vector<Rectangle> Block;
typedef std::pair<Block, Block> Kernel;
typedef std::pair<std::vector<Quad>, std::vector<Quad> > CompiledKernel;


///This function computes the sum of the pixels under the rectangles
///specified in v. Overlapping rectangles will cause the pixels to be
///added in multiple times.
///@param v The list of rectangles.
///@param p The centre point of the rectangles.
///@return The sum of the pixels under the rectangles.
template<class C> inline C integral_rect_sum(const std::vector<Quad>& v, const C* p)
{
	C sum=0; 

	for(unsigned int j=0; j < v.size(); j++)
		sum += p[v[j].topleft] + p[v[j].bottomright] - p[v[j].topright] - p[v[j].bottomleft];
	
	return sum;
}


template<class C> inline C apply_kernel(const CompiledKernel& k, const C* p)
{
	return  integral_rect_sum(k.first, p) - integral_rect_sum(k.second, p);
}

//template<class C> void apply_kernel_to_image(const CVD::BasicImage<C>& integral_image, const Kernel& k, CVD::BasicImage<C>& out);

void apply_kernel(const CVD::BasicImage<float>& in, const Kernel& k, CVD::BasicImage<float>& out);

//void apply_kernel_to_image(const CVD::BasicImage<float>& integral_image, const Kernel& k, CVD::BasicImage<float>& out);

int extent(const Kernel& k);

CVD::Image<int> viola_jones(const CVD::BasicImage<int> in, long seed, float ss, float ns);
CompiledKernel compile_kernel(const Kernel& k, int w);

std::istream& operator>>(std::istream& i, Kernel& k);
std::ostream& operator<<(std::ostream& o, const Kernel& k);

struct parse_error{};
#endif
