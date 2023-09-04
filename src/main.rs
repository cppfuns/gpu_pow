extern crate cuda_sys as cuda;
use std::ffi::CString;

fn main() {
    unsafe {
        // 初始化 CUDA
        cuda::cudaInit(0);

        // 获取可用的 GPU 设备数量
        let mut device_count: i32 = 0;
        cuda::cudaGetDeviceCount(&mut device_count);
        println!("Found {} CUDA devices.", device_count);

        // 遍历每个设备
        for device_id in 0..device_count {
            // 选择当前设备
            cuda::cudaSetDevice(device_id);

            // 获取设备属性
            let mut device_prop: cuda::cudaDeviceProp = std::mem::zeroed();
            cuda::cudaGetDeviceProperties(&mut device_prop, device_id);

            let name_cstr = CString::from_raw(&device_prop.name as *const i8 as *mut i8);
            let name = name_cstr.to_str().unwrap();

            println!("Device {}: {}", device_id, name);
            println!("  Compute Capability: {}.{}", device_prop.major, device_prop.minor);
            println!("  Total Global Memory: {} bytes", device_prop.totalGlobalMem);
            println!("  CUDA Cores: {}", device_prop.multiProcessorCount);
            println!("  Clock Rate: {} KHz", device_prop.clockRate);
            println!();
        }

        // 清理 CUDA
        cuda::cudaDeviceReset();
    }
}
