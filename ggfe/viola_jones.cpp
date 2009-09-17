#include <cvd/integral_image.h>
#include <cvd/image_io.h>
//#include <cvd/gl_helpers.h>
#include <sstream>
#include <vector>
#include <set>
#include <iterator>
#include <cstdlib>
#include <utility>

#include "viola_jones.h"

using namespace CVD;
using namespace std;

#ifdef DEBUG_MAXIMA
	#include <tag/fn.h>
	#include <cvd/videodisplay.h>
	using namespace tag;
#endif

////////////////////////////////////////////////////////////////////////////////
///Quads are faster than Rectangles, but the image width must be known.
///This turns a bunch of rectangles in to a bunch of quads.
///@param block The bunch of rectangles.
///@param width The image width.
///@return The bunch of quads.
vector<Quad> compile_block_to_quads(const Block& block, int width)
{
	vector<Quad> r;
	
	//Precompile the classifier by computing memory offsets.
	for(unsigned int j=0; j < block.size(); j++)
	{
		ImageRef tl =  block[j].first, br =  block[j].second;
		Quad q;

		q.topleft    = tl.y * width + tl.x;
		q.topright   = tl.y * width + br.x; 
		q.bottomleft = br.y * width + tl.x;
		q.bottomright= br.y * width + br.x;

		r.push_back(q);
	}
	
	return r;
}

///Compute the square, symmetric extent of a kernel.
///@param k The kernel.
///@return the extent
int extent(const Kernel& k)
{
	//Compute the offsets in the kernel. This determines the border
	//where the kernel can not be computed
	int max_offset = 0;

	for(unsigned int i=0; i < k.first.size(); i++)
	{
		max_offset = max(max_offset, abs(k.first[i].first.x));
		max_offset = max(max_offset, abs(k.first[i].first.y));
		max_offset = max(max_offset, abs(k.first[i].second.x));
		max_offset = max(max_offset, abs(k.first[i].second.y));
	}
	for(unsigned int i=0; i < k.second.size(); i++)
	{
		max_offset = max(max_offset, abs(k.second[i].first.x));
		max_offset = max(max_offset, abs(k.second[i].first.y));
		max_offset = max(max_offset, abs(k.second[i].second.x));
		max_offset = max(max_offset, abs(k.second[i].second.y));
	}
	//	cout << "MAX OFFSET " << max_offset << endl;
	return max_offset;
}

CompiledKernel compile_kernel(const Kernel& k, int w)
{
	return make_pair(compile_block_to_quads(k.first, w), compile_block_to_quads(k.second, w));
}

///This takes kernel and applies it to an integral image. The input and output images must
///be the same size.
///@param integral_image The integral image
///@param kernel The kernel to apply
///@param out The output image
template<class C> void apply_kernel_to_image(const BasicImage<C>& integral_image, const Kernel& k, BasicImage<C>& out)
{
	//The input and output images must be the same size
	assert(integral_image.size() == out.size());

	//Compile the blocks to a faster version. The fast versions use
	//integer pointer offsets, so the image width must be known.
	vector<Quad> c1 = compile_block_to_quads(k.first, integral_image.row_stride());
	vector<Quad> c2 = compile_block_to_quads(k.second, integral_image.row_stride());

	int max_offset = extent(k);

	//Apply the kernel to the image
	for(int y=max_offset; y < integral_image.size().y - max_offset; y++)
		for(int x=max_offset; x < integral_image.size().x - max_offset; x++)
		{
			const C* p = &integral_image[y][x];
			out[y][x] = integral_rect_sum(c1, p) - integral_rect_sum(c2, p);
		}

}

void apply_kernel(const CVD::BasicImage<float>& in, const Kernel& k, CVD::BasicImage<float>& out) {
  apply_kernel_to_image(in, k, out);
}

ostream& operator<<(ostream& o, const Rectangle& p)
{
	o << '[' << p.first.x << "," << p.first.y << "," << p.second.x << "," << p.second.y << ']';
	return o;
}

ostream& operator<<(ostream& o, const Block& b)
{
	o << '(';
		for(unsigned int i=0; i < b.size(); i++)
			o << b[i];
	o << ')';
	return o;
}

ostream& operator<<(ostream& o, const Kernel& k)
{
	o << '{' << k.first << k.second << '}';
	return o;
}

struct match
{
	char c;
	match(char cc):c(cc){}
};


istream& operator>>(istream& i, const match& m)
{
	i >> ws;
	if(i.get() != m.c)
		throw parse_error();

	return i;
}

char peek(istream& i)
{
	i >> ws;
	return i.peek();
}

istream& operator>>(istream& i, Rectangle& p)
{
	i>> match('[');
	i >> p.first.x >> match(',') >> p.first.y >> match(',') >> p.second.x >> match(',') >> p.second.y;
	i>>match(']');
	return i;
}

istream& operator>>(istream& i, Block& b)
{
	i >> match('(');
	while(peek(i) == '[')
	{
		Rectangle r;
		i >> r;
		b.push_back(r);
	}
	i >> match(')');
	return i;
}


istream& operator>>(istream& i, Kernel& k)
{

	return i >> match('{') >> k.first >> k.second >> match('}');
}
