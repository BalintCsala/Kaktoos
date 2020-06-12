#include <cstdint>
#include <memory.h>
#include <cstdio>
#include <ctime>
#include <thread>
#include <vector>
#include <mutex>
#include <chrono>

#define RANDOM_MULTIPLIER 0x5DEECE66DULL
#define RANDOM_ADDEND 0xBULL
#define RANDOM_MASK ((1ULL << 48ULL) - 1ULL)

#ifndef FLOOR_LEVEL
#define FLOOR_LEVEL 63LL
#endif

#ifndef WANTED_CACTUS_HEIGHT
#define WANTED_CACTUS_HEIGHT 8ULL
#endif

#ifndef WORK_UNIT_SIZE
#define WORK_UNIT_SIZE (1ULL << 23ULL)
#endif

#ifndef WORK_RANDOM_SIZE
#define WORK_RANDOM_SIZE (WORK_UNIT_SIZE + 1000)
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024ULL
#endif

#ifndef GPU_COUNT
#define GPU_COUNT 1ULL
#endif

#ifndef OFFSET
#define OFFSET 0
#endif

#ifndef END
#define END (1ULL << 48ULL)
#endif


__device__ inline int8_t extract(const int8_t heightMap[], uint32_t i) {
    return (int8_t) (heightMap[i >> 1ULL] >> ((i & 1ULL) << 2ULL)) & 0xF;
}

__device__ inline void increase(int8_t heightMap[], uint32_t i) {
    heightMap[i >> 1ULL] += 1ULL << ((i & 1ULL) << 2ULL);
}

namespace java_random {

    // Random::next(bits)
    __device__ inline uint32_t next(uint64_t *random, size_t *i, int32_t bits) {
        return (uint32_t) (random[++*i] >> (48ULL - bits));
    }

    __device__ inline int32_t next_int_unknown(uint64_t *random, size_t *i, int16_t bound) {
        if ((bound & -bound) == bound) {
            return (int32_t) ((bound * (random[++*i] >> 17ULL)) >> 31ULL);
        }

        int32_t bits, value;
        do {
            bits = random[++*i] >> 17ULL;
            value = bits % bound;
        } while (bits - value + (bound - 1) < 0);
        return value;
    }

    // Random::nextInt(bound)
    __device__ inline uint32_t next_int(uint64_t *random, size_t *i) {
        return java_random::next(random, i, 31) % 3;
    }

}

struct rand_params {
	uint64_t multiplier;
	uint64_t addend;
};

__host__ uint64_t get_start_seed(uint64_t offset) {
	uint64_t seed = 0;
	uint64_t i = 0;
	for (; i + 1024 * 1024 <= offset; i += 1024 * 1024) {
		seed = (seed * 280364631195649ULL + 215216710221824ULL) & RANDOM_MASK;
	}
	for (; i < offset; i++) {
		seed = (seed * RANDOM_MULTIPLIER + RANDOM_ADDEND) & RANDOM_MASK;
	}
	return seed;
}

__host__ void gen_rand_params(rand_params *rp, size_t n) {
	rp[0].multiplier = 1;
	rp[0].addend = 0;
	for (size_t i = 1; i < n; i++) {
		rp[i].multiplier = (rp[i - 1].multiplier * RANDOM_MULTIPLIER) & RANDOM_MASK;
		rp[i].addend = (rp[i - 1].addend * RANDOM_MULTIPLIER + RANDOM_ADDEND) & RANDOM_MASK;
	}
}

// TODO: generate low bits seperately so crack() only has to index 32 bit values, ~3% speed improvement
__global__ __launch_bounds__(BLOCK_SIZE, 2) void init(rand_params *rp, uint64_t *random, uint64_t seed, uint64_t m, uint64_t a) {
	size_t index = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;
	seed = (seed * rp[index].multiplier + rp[index].addend) & RANDOM_MASK;
	random[index] = seed;
	for (size_t i = index + stride; i < WORK_RANDOM_SIZE; i += stride) {
		seed = (seed * m + a) & RANDOM_MASK;
		random[i] = seed;
	}
}

__global__ __launch_bounds__(BLOCK_SIZE, 2) void crack(uint64_t *random, int32_t *num_seeds, uint64_t *seeds) {
    size_t seedIndex = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t originalSeed = random[seedIndex];

    int8_t heightMap[512];

#pragma unroll
    for (int i = 0; i < 512; i++) {
        heightMap[i] = 0;
    }

    int16_t currentHighestPos = 0;
    int16_t terrainHeight;
    int16_t initialPosX, initialPosY, initialPosZ;
    int16_t posX, posY, posZ;
    int16_t offset, posMap;

    int16_t i, a, j;

    for (i = 0; i < 10; i++) {
        // Keep, most threads finish early this way
        if (WANTED_CACTUS_HEIGHT - extract(heightMap, currentHighestPos) > 9 * (10 - i))
            return;

        initialPosX = java_random::next(random, &seedIndex, 4) + 8;
        initialPosZ = java_random::next(random, &seedIndex, 4) + 8;
        terrainHeight = (extract(heightMap, initialPosX + initialPosZ * 32) + FLOOR_LEVEL + 1) * 2;

        initialPosY = java_random::next_int_unknown(random, &seedIndex, terrainHeight);

        for (a = 0; a < 10; a++) {
            posX = initialPosX + java_random::next(random, &seedIndex, 3) - java_random::next(random, &seedIndex, 3);
            posY = initialPosY + java_random::next(random, &seedIndex, 2) - java_random::next(random, &seedIndex, 2);
            posZ = initialPosZ + java_random::next(random, &seedIndex, 3) - java_random::next(random, &seedIndex, 3);

            posMap = posX + posZ * 32;
            // Keep
            if (posY <= extract(heightMap, posMap) + FLOOR_LEVEL && posY >= 0)
                continue;

            offset = 1 + java_random::next_int_unknown(random, &seedIndex, java_random::next_int(random, &seedIndex) + 1);

            for (j = 0; j < offset; j++) {
                if ((posY + j - 1) > extract(heightMap, posMap) + FLOOR_LEVEL || posY < 0) continue;
                if ((posY + j) <= extract(heightMap, (posX + 1) + posZ * 32) + FLOOR_LEVEL && posY >= 0) continue;
                if ((posY + j) <= extract(heightMap, posX + (posZ - 1) * 32) + FLOOR_LEVEL && posY >= 0) continue;
                if ((posY + j) <= extract(heightMap, (posX - 1) + posZ * 32) + FLOOR_LEVEL && posY >= 0) continue;
                if ((posY + j) <= extract(heightMap, posX + (posZ + 1) * 32) + FLOOR_LEVEL && posY >= 0) continue;

                increase(heightMap, posMap);

                if (extract(heightMap, currentHighestPos) < extract(heightMap, posMap)) {
                    currentHighestPos = posMap;
                }
            }
        }

        if (extract(heightMap, currentHighestPos) >= WANTED_CACTUS_HEIGHT) {
            seeds[atomicAdd(num_seeds, 1)] = originalSeed;
            return;
        }
    }
}


struct GPU_Node {
    int *num_seeds;
    uint64_t *seeds;
	uint64_t *random;
	rand_params *rp;
};

void setup_gpu_node(GPU_Node *node, int32_t gpu) {
    cudaSetDevice(gpu);
    cudaMallocManaged(&node->num_seeds, sizeof(*node->num_seeds));
    cudaMallocManaged(&node->seeds, 1ULL << 10ULL);
    cudaMallocManaged(&node->random, WORK_RANDOM_SIZE * sizeof(*node->random));
    cudaMallocManaged(&node->rp, WORK_RANDOM_SIZE * sizeof(*node->rp));
}

GPU_Node nodes[GPU_COUNT];
uint64_t offset = OFFSET;
uint64_t seed = get_start_seed(OFFSET);
uint64_t count = 0;
std::mutex info_lock;
std::vector<uint64_t> seeds;
rand_params *rp;

void gpu_manager(int32_t gpu_index) {
    std::string fileName = "kaktoos_seeds" + std::to_string(gpu_index) + ".txt";
    FILE *out_file = fopen(fileName.c_str(), "w");
	cudaSetDevice(gpu_index);
	for (size_t i = 0; i < WORK_RANDOM_SIZE; i++)
		nodes[gpu_index].rp[i] = rp[i];
    while (offset < END) {
        *nodes[gpu_index].num_seeds = 0;
		uint64_t m = rp[WORK_UNIT_SIZE].multiplier, a = rp[WORK_UNIT_SIZE].addend;
        init<<<WORK_UNIT_SIZE / BLOCK_SIZE, BLOCK_SIZE, 0>>>(nodes[gpu_index].rp, nodes[gpu_index].random, seed, m, a);
        info_lock.lock();
		seed = (seed * m + a) & RANDOM_MASK;
		offset += WORK_UNIT_SIZE;
        info_lock.unlock();
        cudaDeviceSynchronize();
        crack<<<WORK_UNIT_SIZE / BLOCK_SIZE, BLOCK_SIZE, 0>>>(nodes[gpu_index].random, nodes[gpu_index].num_seeds, nodes[gpu_index].seeds);
        cudaDeviceSynchronize();
        for (int32_t i = 0, e = *nodes[gpu_index].num_seeds; i < e; i++) {
            fprintf(out_file, "%lld\n", (long long int) nodes[gpu_index].seeds[i]);
            seeds.push_back(nodes[gpu_index].seeds[i]);
        }
        fflush(out_file);
        info_lock.lock();
        count += *nodes[gpu_index].num_seeds;
        info_lock.unlock();
    }
    fclose(out_file);
}


int main() {
	printf("Searching %ld total seeds...\n", (long int) (END - OFFSET));
	
	rp = new rand_params[WORK_RANDOM_SIZE];
	gen_rand_params(rp, WORK_RANDOM_SIZE);
    std::thread threads[GPU_COUNT];

    time_t startTime = time(nullptr), currentTime;
    for (int32_t i = 0; i < GPU_COUNT; i++) {
        setup_gpu_node(&nodes[i], i);
        threads[i] = std::thread(gpu_manager, i);
    }

    using namespace std::chrono_literals;

    while (offset < END) {
        time(&currentTime);
        int timeElapsed = (int) (currentTime - startTime);
        double speed = (double) (offset - OFFSET) / (double) timeElapsed / 1000000.0;
        printf("Searched %lld seeds, offset: %lld found %lld matches. Time elapsed: %ds. Speed: %.2fm seeds/s. %f%%\n",
               (long long int) (offset - OFFSET),
               (long long int) offset,
               (long long int) count,
               timeElapsed,
               speed,
               (double) (offset - OFFSET) / (END - OFFSET) * 100);

        if (timeElapsed % 2000 == 0) {
            printf("Backup seed list:\n");
            for (auto &seed : seeds) {
                printf("%llu\n", (unsigned long long) seed);
            }
        }

        std::this_thread::sleep_for(1s);
    }

    for (auto &thread : threads) {
        thread.join();
    }

    printf("Done!\n");
    printf("But, verily, it be the nature of dreams to end.\n");
}