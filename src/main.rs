extern crate rust_cuda;
extern crate rayon;

use rust_cuda::prelude::*;
use rayon::prelude::*;
use std::time::{Instant, Duration};

const N: usize = 1_000_000; // 向量的大小
const NUM_GPUS: usize = 2; // 使用的显卡数量

fn main() {
    // 初始化 CUDA 上下文
    let contexts: Vec<cuda::Context> = (0..NUM_GPUS)
        .map(|i| cuda::Context::new(i as i32))
        .collect();

    // 创建 CUDA 流
    let streams: Vec<cuda::Stream> = contexts.iter().map(|context| cuda::Stream::new(context)).collect();

    // 创建输入向量和输出向量
    let mut a = vec![1.0; N];
    let mut b = vec![2.0; N];
    let mut c = vec![0.0; N];

    // 使用 Rayon 并行计算框架，将任务分发到多个 GPU 设备
    let start_time = Instant::now();

    (0..NUM_GPUS).into_par_iter().for_each(|i| {
        // 获取当前 GPU 的上下文和流
        let context = &contexts[i];
        let stream = &streams[i];

        // 计算每个 GPU 的任务范围
        let chunk_size = N / NUM_GPUS;
        let start = i * chunk_size;
        let end = start + chunk_size;

        // 分配 GPU 内存并传输数据到 GPU
        let a_device = cuda::Memory::from_slice(&a[start..end], stream);
        let b_device = cuda::Memory::from_slice(&b[start..end], stream);
        let c_device = cuda::Memory::from_mut_slice(&mut c[start..end], stream);

        // 执行向量加法的 CUDA 核心
        cuda::launch!{
            vector_add<<<1, 256, 0, stream.clone()>>>(
                a_device.as_device_ptr(),
                b_device.as_device_ptr(),
                c_device.as_device_mut_ptr(),
                chunk_size as i32,
            )
        };

        // 同步 CUDA 流
        cuda::stream_sync(stream);
    });

    // 停止计时器并输出执行时间
    let end_time = Instant::now();
    let elapsed_time = end_time - start_time;
    println!("GPU execution time: {:?}", elapsed_time);

    // 验证 GPU 计算的结果
    for i in 0..N {
        assert_eq!(c[i], a[i] + b[i]);
    }

    println!("GPU computation verified.");
}

// CUDA 核心，执行向量加法
#[no_mangle]
extern "C" fn vector_add(a: *const f32, b: *const f32, c: *mut f32, n: i32) {
    let index = cuda::block_idx_x() * cuda::block_dim_x() + cuda::thread_idx_x();
    if index < n {
        unsafe {
            *c.offset(index as isize) = *a.offset(index as isize) + *b.offset(index as isize);
        }
    }
}