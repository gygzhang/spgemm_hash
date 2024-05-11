#include <hip/hip_runtime.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/scan.h>
#include"logger.hpp"

typedef csrIdxType idxType;

// class CSR{
// public:
// 	CSR(const size_t _m, const size_t _n, const size_t _nnz, const csrIdxType* _dptr_offset, const csrIdxType* _dptr_colindex, const dtype* _dptr_value)
// 	: m(_m), n(_n), nnz(_nnz), dptr_offset(_dptr_offset), dptr_colindex(_dptr_colindex), dptr_value(_dptr_value)
// 	{}

// 	const size_t m; 
// 	const size_t n;
// 	const size_t nnz;
// 	const csrIdxType* drptr;
// 	const csrIdxType* dcids;
// 	const dtype* dval;
// };

template<const uint NT, const uint TBSIZE>
__global__ void spgemm1(const idxType* d_arptr, const idxType* d_acids,
						const idxType* d_brptr, const idxType* d_bcids,
						idxType *d_row_nz, idxType am
){
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	uint rid = i/NT;
	uint cid = i%NT;

	//TBSIZE/blockDim.x
	__shared__ int s_table[TBSIZE];
	for(int i=cid; i<TBSIZE/blockDim.x*4; i+=NT){
		s_table[i] = -1;
	}
	assert(TBSIZE/blockDim.x*4==32);

}


// void  stat(CSR &A, CSR &B, CSR &C)
// {
// 	logX(A.m,A.n, A.nnz);
// 	logX(B.m,B.n, B.nnz);
// 	vector<int> nnz_per_row_A(A.m, 0); 
// 	int sum= 0;
// 	vector<int> bin(10,0);
// 	int max_nnz_row = 0;
// 	for(int i=0; i<A.m; i++){
// 		nnz_per_row_A[i] = A.drptr[i+1]-A.drptr[i];
// 		max_nnz_row = max(max_nnz_row, nnz_per_row_A[i]);
// 		if(nnz_per_row_A[i]==0) continue;
// 		else if(nnz_per_row_A[i]<32) bin[0]++;
// 		else if(nnz_per_row_A[i]<64) bin[1]++;
// 		else if(nnz_per_row_A[i]<128) bin[2]++;
// 		else if(nnz_per_row_A[i]<256) bin[3]++;
// 		else bin[4]++;
// 		sum+=nnz_per_row_A[i];
// 	}
// 	assert(sum==A.nnz);
// 	logX(bin[0], bin[1], bin[2], bin[3], bin[4], max_nnz_row);
// 	const int BS = 512;
// 	auto div_up = [&](int nt, int bs) -> int
// 	{
// 		if(nt%bs!=0) return nt/bs+1;
// 		return nt/bs;
// 	};
// 	int GS = div_up(bin[0]*4, BS);
// 	logX(bin[0], GS);
// 	spgemm1<4, 4096> <<< GS, BS >>> (A.drptr, A.dcids, B.drptr, B.daids, );
// }

int div_up(int a, int b)
{
	if((a&(b-1))!=0) return a/b+1;
	return a/b;
}


__global__ void get_row_interim(const csrIdxType* drptrA, const csrIdxType* dcidxA, const csrIdxType* drptrB,
								int *drowinterim, const uint m, int* dtotalinterim, int *dmaxrowinterim)
{

	const uint ith = blockIdx.x*blockDim.x+threadIdx.x;
	if(ith>=m) return;
	int interim = 0;
	// 对于第ith行的每一个数据
	for(int i=drptrA[ith]; i<drptrA[ith+1]; i++){
		// 对于第ith行其中一个数据x， 它所在的列是acol
		const uint acol = dcidxA[i];
		// B的acol行有多少个非0的数据， x就要产生多少个中间结果。
		interim += (drptrB[acol+1]-drptrB[acol]);
	}

	drowinterim[ith] = interim;
	atomicAdd(dtotalinterim, interim);
	atomicMax(dmaxrowinterim, interim);
	// assert(interim<=32);
	// *dtotalinterim =1;
}

__global__ void get_nnzrow_cnt(const csrIdxType* drptrA, int *dnnzrowcnt, int m)
{
	int ith = blockIdx.x*blockDim.x + threadIdx.x;
	if(ith>=m) return;
	// assert(ith<4096&&gridDim.x==16);
	if(drptrA[ith+1]-drptrA[ith]>0) {
		// assert(drptrA[ith+1]-drptrA[ith]<0);
		// atomicAdd(dnnzrowcnt, drptrA[ith+1]-drptrA[ith]);
		atomicAdd(dnnzrowcnt, 1);
	}
	// *dnnzrowcnt = 1;
}

__global__ void set_bucket(int *drowinterim, int *dper_buck_size, const uint m)
{
	const uint ir = blockDim.x*blockIdx.x + threadIdx.x;

	if(ir>=m) return;
	const uint rowiterim = drowinterim[ir];

	if(rowiterim==0) return;

	if(rowiterim<=32){
		atomicAdd(&dper_buck_size[0], 1);
		// atomicAdd(dbucket[0]+idx, ir);
	}else if(rowiterim<=512){
		atomicAdd(&dper_buck_size[1], 1);
		// atomicAdd(dbucket[1]+idx, ir);
	}else if(rowiterim<=1024){
		atomicAdd(&dper_buck_size[2], 1);
		// atomicAdd(dbucket[2]+idx, ir);
	}else{
		atomicAdd(&dper_buck_size[3], 1);
		// atomicAdd(dbucket[3]+idx, ir);
	}
}

__global__ void set_bucket_item(int *drowinterim, int *dper_buck_size, int **dbuckets, const uint m)
{
	const uint ir = blockDim.x*blockIdx.x + threadIdx.x;
	if(ir>=m) return;
	const uint rowiterim = drowinterim[ir];
	// __shared__ int temp[4];
	// if(ir<4) temp[ir]=0;
	// __syncthreads();
	if(rowiterim==0) return;
	if(rowiterim<=32){
		uint idx = atomicAdd(&dper_buck_size[0], 1);
		*(dbuckets[0]+idx) = ir;
	}else if(rowiterim<=512){
		uint idx = atomicAdd(&dper_buck_size[1], 1);
		*(dbuckets[1]+idx) = ir;
	}else if(rowiterim<=1024){
		uint idx = atomicAdd(&dper_buck_size[2], 1);
		*(dbuckets[2]+idx) = ir;
	}else{
		uint idx = atomicAdd(&dper_buck_size[3], 1);
		*(dbuckets[3]+idx) = ir;
	}
	// __syncthreads();
	// if(ir<4) dper_buck_size[ir] = temp[ir];
	

}

uint hash32(const uint x){
	return (x*51)&(32-1);
}



template<const uint NT, const uint LDSSIZE, const uint MAXINTERIMM>
__global__ void hash_symbolic_nt(  const csrIdxType* drptrA, const csrIdxType* dcidxA, 
								const csrIdxType* drptrB,	const csrIdxType* dcidxB, 
								int *dbucket ,int *dresnnzperrow, const uint nrow, int *drestotal
								)
{
	const uint gidx = blockIdx.x*blockDim.x + threadIdx.x;
	
	// 把所有线程按4个一行的组织起来
	int rid = gidx/NT;
	const uint cid = gidx%NT;
	// 4个线程处理1行， 一共256个线程， 那么一共要处理256/4=64行， 一行最多32个中间结果，那么LDS大小应该为64*32
	// 每4个线程要处理A的一行，假设A一行产生的interim不超过MAXINTERIMM， 那么LDSSIZE应该等于 blockDim.x*MAXINTERIMM/NT =  256*32/4
	// assert(blockDim.x*MAXINTERIMM/NT == 256*32/4);
	// s_table是用来
	__shared__ int s_table[LDSSIZE];
	// 一个block里面有blockDim.x个线程，每个NT个线程处理一行， 那么一个block可以处理blockDim.x/NT行
	// 因为LDS是在一个block内共享的，要计算一个block内有每个线程组（NT=4）应该， 对于一个block， 其中一个block
	assert(gridDim.x>1);
	// 每4个线程计算一行，一行会产生最多MAXINTERIMM个中间结果，因此s_table的前32个位置代表这32个中间结果的列的位置，
	// 如果s_table对应位置没有产生结果， 那么应该将这个位置设置成-1，不为-1代表，C[rid][key]有结果。
	// block内的偏移， 4个线程共用MAXINTERIMM大小的位置, 先将一个block内的线程分成4个一行，那么一个block内有blockDim.x/NT行
	// 
	const uint rowinterimmoffset = (rid%(blockDim.x/NT))*MAXINTERIMM;
	for(int i=cid; i<MAXINTERIMM; i+=NT){
		s_table[rowinterimmoffset+i] = -1;
	}
	if(rid>=nrow) return;

	rid = dbucket[rid];
	int row_nz=0;

	// 四个线程处理一行， 也就是一行里面的连续的4个元素由4个连续的线程处理，这4个线程rid相同，但是cid不同。
	for(int i = drptrA[rid]+cid; i<drptrA[rid+1]; i+=NT){
		// 某个线程负责的A中的元素的列下标
		const uint acidx =dcidxA[i];
		// 现在每个线程负责处理B中第acidx行
		for(int j = drptrB[acidx]; j<drptrB[acidx+1]; j++){
			// 在B中的列下标
			const uint bcol = dcidxB[j];
			int key = bcol;
			int hash =  (bcol*107)&(32-1);
			int hidx = rowinterimmoffset + hash;

			// assert(bcol<0);
			while(1){
				// assert(s_table[hidx]!=-1);
				// 只需要知道结果矩阵中(rid, bcol)不为0，我们需要这个信息来决定要不要为这个位置分配内存，计算留到numuric阶段
				if(s_table[hidx]==key) break;
				// 如果没有被其他线程更新过
				else if(s_table[hidx]==-1){
					// if s_table[hidx]==-1; then s_table[hidx] = bcol;
					// 防止一个rid内的多个线程同时更新同一个位置，需要原子操作
					int old = atomicCAS(s_table+hidx, -1, key);
					// 只有一个线程能执行这一句
					if(old==-1){
						row_nz++;
						break;
					}
				// 发生了哈希冲突， 采用线性探测法 
				}else{
					hash = (hash+1)&(32-1);
					hidx = rowinterimmoffset + hash;
				}
			}
		}
	}

	// if(row_nz!=0) assert(row_nz==0);
	// 现在每个线程手上可能都在s_table上更新了一些数据
	// atomicAdd(drestotal, row_nz);
	// __syncthreads();
	// 现在将同一个warp内的数据加起来
	// NT=4, 所以两次洗牌就可以了
	row_nz += __shfl_xor(row_nz, 2);
	row_nz += __shfl_xor(row_nz, 1);
	__syncthreads();
	// if(row_nz!=0) assert(row_nz==0);
	// 第一列的线程将结果写回到内存。其他列也行，因为现在每一列的数据都是一样的
	// row_nz代表结果矩阵一行有多少个非零的结果
	if(cid==0){
		dresnnzperrow[rid] = row_nz;
	}

}

// #define SHL(x, s) ((unsigned int) ((x) << ((s) & 31)))
// #define SHR(x, s) ((unsigned int) ((x) >> (32 - ((s) & 31))))
// #define ROTL(x, s) ((unsigned int) (SHL((x), (s)) | SHR((x), (s))))
#define ROTL(x, b) (uint)(((x) << (b)) | ((x) >> (32 - (b))))

template<const uint NT, const uint LDSSIZE, const uint MAXINTERIMM>
__global__ void hash_numuric_nt(  const csrIdxType* drptrA, const csrIdxType* dcidxA, const dtype *dvalA,
								const csrIdxType* drptrB,	const csrIdxType* dcidxB, const dtype *dvalB,
								const csrIdxType* drptrC,	csrIdxType* dcidxC, dtype *dvalC,
								int *dbucket ,int *dresnnzperrow, const uint nrow, int *drestotal,  int *dresperrowcnt, double alpha
								)
{
	uint gidx = blockIdx.x*blockDim.x +  threadIdx.x;
	uint rid = gidx/NT;
	uint cid = gidx%NT;
	// 一个block 处理256/4行数据，每行最多产生8个数据， 那么8*256/4应该够用了？？？？
	// __shared__ int s_table[32*256/4];
	__shared__ int s_table[2048];
	__shared__ double v_table[2048];
	// assert(blockDim.x*8)
	// assert(rid%(blockDim.x/4)<64);
	// assert((blockIdx.x/4)<64);
	assert((threadIdx.x/4)==rid%(blockDim.x/4)); 
	assert(blockDim.x*8==2048);
	// int offset = (threadIdx.x/4)*32;
	// blockDim.x/4代表将一个block内的线程组织成4个一行， 0123， 4567...这样
	// 同样， offset代表每个rid写入位置的起始下标， 比如rid=0的4个线程写s_table[0:32], rid=1则写s_table[32:64]，依次类推  
	int offset = (rid%(blockDim.x/4))*32;
	for(int i=cid; i<32; i+=4){
		s_table[offset+i] = -1;
		v_table[offset+i] = 0.;
	}
	// if(cid==0){
	// 	for(int i=0; i<2048; i++) {s_table[i] = -1; v_table[i] = 0.;}
	// }
	__syncthreads();

	// if(cid==0){
	// 	for(int i=0; i<2048; i++) assert(v_table[i]==0);
	// }
	// return;
	

	// 有结果的行只有nrow行， 因此只需要nrow个nt-线程组
	// 和reduce优化里面一样， 使用具有连续线程id的线程来处理不连续的行(因为有的行可能全为0)
	if(rid>=nrow) return;
	// rid现在是在A中的行id
	rid = dbucket[rid];
	if(cid==0) dresperrowcnt[rid]=0;
	// 之前求的前缀和派上用场了， rptrC_offset既代表第rid行前面的行总共有多少个非零值，也代表本行的开始写入下标
	int rptrC_offset = drptrC[rid];
	for(int i=drptrA[rid]+cid; i<drptrA[rid+1]; i+=4){
		// 同一个工作组中的4个线程都在A的一行中拿到自己对应的数据
		int acol = dcidxA[i];
		double aval = dvalA[i];
		// 现在每一个线程要开始属于自己的循环， 这个循环是在B的一行上进行的
		// 哪一行呢？ 对于A中一行的数据，把rid行顺时针旋转90度， 现在cid负责的那一个数据在哪一行， 就循环遍历B中哪一行。
		for(int j=drptrB[acol]; j<drptrB[acol+1]; j++){

			int bcol = dcidxB[j];
			// assert(bcol<4&&bcol>=0);
			double bval = dvalB[j];
			int hash = (bcol*251)&(32-1);
			int hidx = offset+hash;
			// assert(hidx<2048);

			while(1){
				// static int iii=0;iii++;
				// if(iii==1) break;
				// assert(hidx<2048);
				// 如果C(rid, bcol)已经被写入过了，那直接将结果累加到C(rid, bcol)上就行
				if(s_table[hidx]==bcol){
					// return;
					atomicAdd(&v_table[hidx], alpha*aval*bval);
					// v_table[cid] += alpha*aval*bval;
					break;
				}
				// 但是如果C(rid, bcol)还没有被写过
				else if(s_table[hidx]==-1){
					// return;
					// assert(hidx<2048);
					int old = atomicCAS(s_table+hidx, -1, bcol);
					// 为什么只让old==-1的线程写入呢？
					// 因为可能会发生哈希冲突，也就是，C中同一行的不同列映射到同一个hidx，
					// 这两个不同列的结果肯定不能写到同一个位置, 必须进行再探测
					if(old==-1){
						// if(hidx>=2048) break;
						atomicAdd(&v_table[hidx], alpha*aval*bval);
						// v_table[cid] += alpha*aval*bval;
						// atomicAdd(v_table+0, 1);
						// return;
						break;
					}
					// 发现hidx被别的线程写了，且和自己bcol映射到同一个slot
					else if(s_table[hidx]==bcol){
						atomicAdd(&v_table[hidx], alpha*aval*bval);
						// v_table[cid] += alpha*aval*bval;
						break;
					}
					// 发现这个slot被占用，且占用的列号还和自己的不一样
					else if(s_table[hidx]!=bcol){
						hash = ((hash+1))&(32-1);
						hidx = offset+hash;
					}
				}else{
					// return;
					// break;
					hash = ((hash+1))&(32-1);
					hidx = offset+hash;
				}
			}
		}
	}
	__syncthreads();
	// 一行最多产生32个结果， 之前这32个位置， 其中可能会有位置没有结果， 现在让所有被更新的元素左对齐
	for(int i=cid; i<32; i+=4){
		if(s_table[offset+i]!=-1){
			int idx = atomicAdd(dresperrowcnt+rid, 1);
			s_table[offset+idx] = s_table[offset+i];
			v_table[offset+idx] = v_table[offset+i];
		}
	}
	__syncthreads();
	
	int nz = dresperrowcnt[rid];
	// assert(nz<=4);
	int cnt, target;
	// 一个nt工作组负责C中的一行， 工作组中的4个线程依次处理不同列
	for(int ii=cid; ii<nz; ii+=4){
		target = s_table[offset+ii];
		//第i个线程的数据应该放到ccidx列， 但是这个列是稠密矩阵的列
		// s_table中存的是bcol， 也是对应v_table中元素在C中存放的colidx
		// C的rid行总共要写nz个元素，他们的列号为s_table[offset: offset+nz]， 不排序随便存，反正一对一
		// 排序的话，列号最小的元素应该写到C rid行的第一个位置， 最大的元素写到第nz个位置。
		// 因此，对于任意一个列号，需要找到有多少个列号比自己小， 知道之后就可以将他们写到他们后面。
		int ccidx = s_table[offset+ii];
		cnt=0;
		// s_table[offset: offset+32]中储存了一行的不同列坐标，对于其中一个 
		for(int j=0; j<nz; j++){
			
			// if(s_table[offset+ii]>s_table[offset+j]) cnt = cnt +1;
			// 如果大于0，最高位是0， 如果小于零，最高位是1， 这是统计比自己小列号的个数
			cnt += uint(s_table[offset+j]-ccidx)>>31;
			// else cnt += ROTL(val,31);
		}
		// if(rid==0 && cnt>0) assert(v_table[offset+i]!=0);
		// if(rid==0) assert(v_table[0]!=0||v_table[1]!=0||v_table[2]!=0);
		// dvalC[rptrC_offset+0] = v_table[offset+ii];
		// dcidxC[rptrC_offset+0] = s_table[offset+ii];
		// 我的列号是ccidx， 我的值是v_table[offset+ii]， 有cnt个数比自己小，那么就将自己的数据写到第cnt个位置
		dvalC[rptrC_offset+cnt] = v_table[offset+ii];
		dcidxC[rptrC_offset+cnt] = ccidx;
		// dvalC[0] = v_table[offset+ii];
		// dcidxC[0] = s_table[offset+ii];
		// int rcnt = drptrC[rid+1]-drptrC[rid];
		// assert(nz==(drptrC[rid]-drptrC[rid-1]));
		
	}
	

	// if(cid==0){
	// 	int rcnt = nz;
	// 	for(int k=0; k<rcnt; k++) {
	// 		dvalC[rptrC_offset+k] = 0.1+k/2.;
	// 	}
	// }
	// assert(nz<=32);
	// assert(rptrC_offset+nz<=3463);
	// for(int i=cid; i<nz; i+=4){
	// 	dvalC[rptrC_offset+i] = v_table[offset+i];
	// 	dcidxC[rptrC_offset+i] = s_table[offset+i];
	// }
	
}

// find num of each row's nnz results
__global__ void hash_symbolic_tb(const csrIdxType* drptrA, const csrIdxType* dcidxA, 
								const csrIdxType* drptrB,	const csrIdxType* dcidxB, 
								int *dbucket ,int *dresnnzperrow, const uint nrow, int *drestotal)
{
	// 现在是每个block处理A的一行
	uint rid = blockIdx.x;
	uint cid = threadIdx.x % 64;
	// 每个warp处理A每一行的一个数据, 
	// TODO: HIP每个warp64个线程， 真的是否高效？
	uint wid = threadIdx.x/64;
	uint wcnt = blockDim.x/64;
	// 现在是每一个block负责一行，s_table的大小表示A的一行在C中产生的结果都不会超过128个， 对于5w规模的数据，应该都不会超过100, 测试是79
	__shared__ int s_table[1024];
	__shared__ int dcrowrestotal[1];
	if(cid==0) dcrowrestotal[0]=0;
	for(int i=threadIdx.x; i<1024; i+=blockDim.x) s_table[i]=-1;

	if(rid>=nrow) return;
	rid = dbucket[rid];
	int crowresnnz = 0;
	// nt版本的应该不用同步，因为warp本身就是同步的
	__syncthreads();
	for(int i=drptrA[rid] + wid; i<drptrA[rid+1]; i+=wcnt){
		int acol = dcidxA[i];
		for(int j=drptrB[acol]+cid; j<drptrB[acol+1]; j+=64){
			int bcol = dcidxB[j]; 
			int hash = (bcol*31)&(1024-1);
			while(1){
				if(s_table[hash]==bcol) break;
				else if(s_table[hash]==-1){
					int old = atomicCAS(s_table+hash, -1, bcol);
					if(old==-1) {
						crowresnnz++;
						break;
					}
				}else{
					hash = (hash+1)&(1024-1);
				}
			}
		}
	}

	// warp reduce
	crowresnnz += __shfl_xor(crowresnnz, 32);
	crowresnnz += __shfl_xor(crowresnnz, 16);
	crowresnnz += __shfl_xor(crowresnnz, 8);
	crowresnnz += __shfl_xor(crowresnnz, 4);
	crowresnnz += __shfl_xor(crowresnnz, 2);
	crowresnnz += __shfl_xor(crowresnnz, 1);
	// 这个同步是用来同步不同warp的，warp是同步的。
	__syncthreads();

	// assert(crowresnnz>0);
	// 不在一个warp内的只能通过共享内存或者全局内存来reduce。
	if(cid==0){
		atomicAdd(dcrowrestotal, crowresnnz);
	}
	// if(crowresnnz!=0) assert(crowresnnz==0);
	__syncthreads();
	// assert(dcrowrestotal[0]==0);17371

	if(threadIdx.x==0) dresnnzperrow[rid] = dcrowrestotal[0];

}
// __launch_bounds__(1024, 2)
template<uint LDSSIZE>
__global__ void  hash_numuric_tb(const csrIdxType* drptrA, const csrIdxType* dcidxA, const dtype *dvalA,
								const csrIdxType* drptrB,	const csrIdxType* dcidxB, const dtype *dvalB,
								const csrIdxType* drptrC,	csrIdxType* dcidxC, dtype *dvalC,
								int *dbucket ,int *dresnnzperrow, const uint nrow, int *drestotal,  int *dresperrowcnt, double alpha)
{
	// 如果一个block128个线程，那么一个block一次可以处理一行的2个元素
	uint rid = blockIdx.x;
	uint wid = threadIdx.x/64;
	uint cid = threadIdx.x%64;
	uint wcnt = blockDim.x/64;
	
	// A一行乘以B的列向量，得到非零值的个数应该小于等于512
	// 64*
	__shared__ int s_table[LDSSIZE];
	__shared__ double v_table[LDSSIZE];
	for(int i=threadIdx.x; i<LDSSIZE; i+=blockDim.x){
		s_table[i] = -1; v_table[i]=0;
	}
	if(rid>=nrow) return;
	rid = dbucket[rid];
	int rptr_offset = drptrC[rid];
	
	// __syncthreads();
	for(int i=drptrA[rid]+wid; i<drptrA[rid+1]; i+=1){
		int acol = dcidxA[i];
		double aval = dvalA[i];
		for(int j=drptrB[acol]+cid; j<drptrB[acol+1]; j+=64){
			int bcol = dcidxB[j];
			double bval = dvalB[j];
			int hash = (bcol)&(LDSSIZE-1);
			while(1){
				if(s_table[hash]==bcol){
					v_table[hash] += alpha * aval*bval;
					// atomicAdd(v_table+hash, alpha * aval*bval);
					break;
				}
				else if(s_table[hash]==-1){
					int old = atomicCAS(s_table+hash, -1, bcol);
					if(old==-1) {
						// nnz++;
						v_table[hash] += alpha * aval*bval;
						// atomicAdd(v_table+hash, alpha * aval*bval);
						break;
					}
				}else{
					// break;
					hash = ((hash+1))&(LDSSIZE-1);
				}
			}
		}
	}
	
	//reduce nnz;
	// __syncthreads();
	int nnz = dresperrowcnt[rid];

	__shared__ int nnz1[1];
	nnz1[0] = 0;
	for(int i=threadIdx.x; i<LDSSIZE; i+=blockDim.x){
		if(s_table[i]!=-1){
			int idx = atomicAdd(nnz1, 1);
			s_table[idx] = s_table[i];
			v_table[idx] = v_table[i];
		}
	}
	nnz = nnz1[0];
	// assert(nnz==nnz1[0]);
	for(int i=threadIdx.x; i<nnz; i+=blockDim.x){
		int cnt = 0;
		// if(s_table[i]==-1) continue;
		for(int j=0; j<nnz; j++){
			// cnt += uint(s_table[j]-s_table[i])>>31;
			if(s_table[i]>s_table[j]) cnt++;
		}
		dcidxC[rptr_offset+cnt] = s_table[i];
		dvalC[rptr_offset+cnt] = v_table[i];
	}


	// if(threadIdx.x==0){
		
	// }
	// __syncthreads();

}

__global__ void test(){

}

#define HIP_ALLOC_MEMSETI(varname, num)      int *d##varname, *h##varname; \
											HIP_CHECK( hipMalloc((void**) &d##varname, num * sizeof(csrIdxType))); \
											HIP_CHECK( hipMemset(d##varname, 0,  num * sizeof(csrIdxType)));	\
											h##varname = new int[num]; 	

#define HIP_ALLOC_MEMSETD(varname, num)      dtype *d##varname, *h##varname; \
											HIP_CHECK( hipMalloc((void**) &d##varname, num * sizeof(dtype))); \
											HIP_CHECK( hipMemset(d##varname, 0,  num * sizeof(dtype)));	\
											h##varname = new dtype[num]; 												
											
#define fori(i, n) for(int i=0; i<n; i++)

void  call_device_spgemm1(
		const dtype alpha, const size_t m, const size_t n, const size_t k,
		const size_t nnz_A, const csrIdxType*  drptrA, const csrIdxType*  dcidxA, const dtype* dvalA,
		const size_t nnz_B, const csrIdxType*  drptrB, const csrIdxType*  dcidxB, const dtype* dvalB,
		size_t* ptr_nnz_C, csrIdxType* drptrC, csrIdxType** pdcidxC, dtype**  pdvalC)
{	
	int *drowinterim, *dtotalinterim, *dnnzrowcnt, *dresnnzperrow, *dmaxrowinterim, *drestotal, *dperbucketsize, **dbucket;
	int *hrowinterim, *htotalinterim, *hresnnzperrow, *hmaxrowinterim, *hrestotal, *hperbucketsize, **hbucket;
	HIP_CHECK( hipMalloc((void**) &drowinterim, m * sizeof(csrIdxType)))
	HIP_CHECK( hipMalloc((void**) &dresnnzperrow, m * sizeof(csrIdxType)))
	HIP_CHECK( hipMalloc((void**) &dtotalinterim, 1 * sizeof(csrIdxType)))
	HIP_CHECK( hipMalloc((void**) &dnnzrowcnt, 1 * sizeof(csrIdxType)))
	HIP_CHECK( hipMalloc((void**) &dmaxrowinterim, 1 * sizeof(csrIdxType)))
	HIP_CHECK( hipMalloc((void**) &drestotal, 1 * sizeof(csrIdxType)))
	HIP_CHECK( hipMalloc((void**) &dperbucketsize, 10 * sizeof(csrIdxType)))
	HIP_CHECK( hipMalloc((void**) &dbucket, 10 * sizeof(csrIdxType*)))

	HIP_CHECK( hipMemset(drowinterim, 0,  m * sizeof(csrIdxType)))
	HIP_CHECK( hipMemset(dresnnzperrow, 0,  m * sizeof(csrIdxType)))
	HIP_CHECK( hipMemset(dnnzrowcnt, 0,  1 * sizeof(csrIdxType)))
	HIP_CHECK( hipMemset(dtotalinterim, 0,  1 * sizeof(csrIdxType)))
	HIP_CHECK( hipMemset(dmaxrowinterim, 0,  1 * sizeof(csrIdxType)))
	HIP_CHECK( hipMemset(drestotal, 0,  1 * sizeof(csrIdxType)))
	HIP_CHECK( hipMemset(dperbucketsize, 0,  10 * sizeof(csrIdxType)))
	HIP_ALLOC_MEMSETI(perbucketsize1, 20);
	
	hrowinterim = new int [m];
	int *nnzrowcnt = new int[1];
	htotalinterim = new int[1];
	hresnnzperrow = new int[m];
	hmaxrowinterim = new int[1];
	hrestotal= new int[1];
	hperbucketsize = new int[10];
	hbucket = new int*[10];
	// for(int i=0; i<m; i++){
	// 	if((drptrA[i+1]-drptrA[i])>0) nnzrowcnt++;
	// }
	const int BS = 256;
	const int GS = div_up(m*4, BS);
	//统计A的中一行的元素不全为0的行数， 即A非0行的行数
	get_nnzrow_cnt<<<GS,BS>>>(drptrA, dnnzrowcnt, m);
	HIP_CHECK( hipMemcpy(nnzrowcnt, dnnzrowcnt, (1) * sizeof(csrIdxType), hipMemcpyDeviceToHost) )
	// 统计A每一行会产生多少个中间结果，以稠密矩阵来说，A一行最多可以产生K*N个中间元素，但最终结果就N个元素
	get_row_interim<<<GS,BS>>>(drptrA, dcidxA, drptrB, drowinterim, m, dtotalinterim, dmaxrowinterim);
	HIP_CHECK( hipMemcpy(hrowinterim, drowinterim, (m) * sizeof(csrIdxType), hipMemcpyDeviceToHost) )
	HIP_CHECK( hipMemcpy(htotalinterim, dtotalinterim, (1) * sizeof(csrIdxType), hipMemcpyDeviceToHost) )
	HIP_CHECK( hipMemcpy(hmaxrowinterim, dmaxrowinterim, (1) * sizeof(csrIdxType), hipMemcpyDeviceToHost) )

	// 根据每一行中间结果的多少，结果为： 中间结果个数小于32个行的个数是x行， 中间结果个数在32到1024的个数是y行。
	set_bucket<<<GS,BS>>>(drowinterim, dperbucketsize, m);

	HIP_CHECK( hipMemcpy(hperbucketsize, dperbucketsize, (10) * sizeof(csrIdxType), hipMemcpyDeviceToHost) );
	logX(hperbucketsize[0], hperbucketsize[1], hperbucketsize[2], hperbucketsize[3], hperbucketsize[4]);

	

	for(int i=0; i<4; i++) HIP_CHECK( hipMalloc((void**) &dbucket[i], hperbucketsize[i] * sizeof(csrIdxType)))
	HIP_CHECK( hipMemset(dperbucketsize, 0,  10 * sizeof(csrIdxType)))
	// 现在知道中间结果在某个区间的行数， 我们就应该把对应行数添加到对应桶中， 也就是说，现在每个桶存的是很多个行的索引下标。
	// 桶的序号越大，其中行产生的中间结果就越多， 这是为了负载均衡。
	set_bucket_item<<<GS,BS>>>(drowinterim, dperbucketsize, dbucket, m);

	// if(*hmaxrowinterim>32){logX(*hmaxrowinterim); exit(-1);}
	logX(*hmaxrowinterim);

	for(int i=0; i<10; i++){
		if(hperbucketsize[i]>0);
	}
	
	HIP_CHECK( hipMemcpy(hbucket, dbucket, (10) * sizeof(csrIdxType), hipMemcpyDeviceToHost) );
	// 现在就可以开始symbolic了。 一共有hperbucketsize[0]行数据，每行数据要4个线程来处理，因此需要hperbucketsize[0]*4个线程
	hash_symbolic_nt<4, 2048, 32> <<<div_up(hperbucketsize[0]*4, BS), BS>>>(drptrA, dcidxA, drptrB, dcidxB, dbucket[0], dresnnzperrow, hperbucketsize[0], drestotal);
	HIP_CHECK( hipMemcpy(hrestotal, drestotal, (1) * sizeof(csrIdxType), hipMemcpyDeviceToHost))
	HIP_CHECK( hipMemcpy(hbucket[0], dbucket[0], (hperbucketsize[0]) * sizeof(csrIdxType), hipMemcpyDeviceToHost))

	hash_symbolic_tb<<<hperbucketsize[1], 64>>>(drptrA, dcidxA, drptrB, dcidxB, dbucket[1], dresnnzperrow, hperbucketsize[1], drestotal);
	HIP_CHECK( hipMemcpy(hresnnzperrow, dresnnzperrow, (m) * sizeof(csrIdxType), hipMemcpyDeviceToHost) )


	HIP_CHECK( hipMemset(drptrC, 0,  (m+1) * sizeof(csrIdxType)))
	thrust::exclusive_scan(thrust::device, dresnnzperrow, dresnnzperrow+m+1, drptrC, 0);

	*ptr_nnz_C = drptrC[m];
	int c_nnz = 0;
	int maxcresnnz = 0;
	// logX(hresnnzperrow[0], hresnnzperrow[1], hresnnzperrow[2]);
	for(int i=0; i<m; i++){
		// drptrC[i] += drptrC[i-1] + dresnnzperrow[i-1];
		
		maxcresnnz = max(maxcresnnz, hresnnzperrow[i]);
	}
	// if(maxcresnnz>32) { logX(maxcresnnz, drptrC[m], hperbucketsize[0]); exit(-1);}
	
	logX(*hmaxrowinterim, maxcresnnz, drptrC[m], hperbucketsize[0], *ptr_nnz_C, *hrestotal);
	// exit(-1);
	HIP_ALLOC_MEMSETI(cidxC, (*ptr_nnz_C)*2);
	HIP_ALLOC_MEMSETD(valC, (*ptr_nnz_C)*2);
	HIP_ALLOC_MEMSETI(rccnt, m);


	hash_numuric_nt<4, 2048, 32> <<<div_up(hperbucketsize[0]*4, BS), BS>>>(drptrA, dcidxA, dvalA, drptrB, dcidxB, dvalB,  \
							drptrC, dcidxC, dvalC, dbucket[0], dresnnzperrow, hperbucketsize[0], drestotal,  drccnt, alpha);

	hash_numuric_tb<512><<<hperbucketsize[1], 64>>>(drptrA, dcidxA, dvalA, drptrB, dcidxB, dvalB,  \
							drptrC, dcidxC, dvalC, dbucket[1], dresnnzperrow, hperbucketsize[1], drestotal,  drccnt, alpha);

	// exit(-1);
	
	HIP_CHECK( hipMemcpy(hvalC, dvalC, (*ptr_nnz_C) * sizeof(double), hipMemcpyDeviceToHost) )
	// logX(*hmaxrowinterim); exit(-1);

	*pdcidxC = dcidxC;
	*pdvalC = dvalC;
	// hperbucketsize1[0] 0~32, hperbucketsize1[1] 32~64, hperbucketsize1[2] 64~128, hperbucketsize1[3] >128
	cout.precision(15);
	// logX(hperbucketsize[0],hperbucketsize[1], hperbucketsize[2], hperbucketsize[3], *hmaxrowinterim, drptrC[m],hvalC[0], hvalC[1]);
	// logX(hvalC[0], hvalC[1], hvalC[3], hvalC[4],hvalC[5], hvalC[6], hvalC[7], hvalC[8]);
	int nnzrowBuck = 0;
	// fori(i, 4){
	// 	int nn = hperbucketsize[i];
	// 	nnzrowBuck += hperbucketsize[i];
	// }

	

	uint sum=0;
	for(int i=0; i<m; i++){
		sum += hrowinterim[i];
	}
	// logX(*hmaxrowinterim); exit(-1);
	// logX(alpha); exit(-1);

	// logX(m, *nnzrowcnt, sum, GS, *htotalinterim, *hmaxrowinterim, *hrestotal, nnzrowBuck, c_nnz);
	// HIP_CHECK( hipMemcpy(dinterim, host_offsetA, (m + 1) * sizeof(csrIdxType), hipMemcpyHostToDevice) )
	hipDeviceSynchronize();
}



void  call_device_spgemm(const int transA,
		const int          transB,
		const dtype        alpha,
		const size_t       m,
		const size_t       n,
		const size_t       k,
		const size_t       nnz_A,
		const csrIdxType*  dptr_offset_A,
		const csrIdxType*  dptr_colindex_A,
		const dtype*       dptr_value_A,
		const size_t       nnz_B,
		const csrIdxType*  dptr_offset_B,
		const csrIdxType*  dptr_colindex_B,
		const dtype*       dptr_value_B,
		size_t*            ptr_nnz_C,
		csrIdxType*        dptr_offset_C,
		csrIdxType**       pdptr_colindex_C,
		dtype**            pdptr_value_C )
{
    // Get the  nonzero num of C and the value in offset_C 
	// CSR A(m, k, nnz_A, dptr_offset_A, dptr_colindex_A, dptr_value_A);
	// CSR B(k, n, nnz_B, dptr_offset_B, dptr_colindex_B, dptr_value_B);
	// CSR C(m, n, nnz_B, dptr_offset_B, dptr_colindex_B, dptr_value_B);
	// stat(A,B,C);


	
	
	

    size_t nonzero = max(nnz_A, nnz_B);
    *ptr_nnz_C = nonzero;

    // Malloc pdptr_colindex_C and pdptr_value_C 
    // HIP_CHECK( hipMalloc((void**) pdptr_colindex_C, nonzero * sizeof(csrIdxType)) )
    // HIP_CHECK( hipMalloc((void**) pdptr_value_C, nonzero * sizeof(dtype)) )
	
	call_device_spgemm1(alpha, m,n,k,nnz_A, dptr_offset_A,dptr_colindex_A, dptr_value_A, 
						nnz_B, dptr_offset_B, dptr_colindex_B, dptr_value_B,ptr_nnz_C, dptr_offset_C, pdptr_colindex_C, pdptr_value_C);

    // Kernels need to implement 
    // ...
}

