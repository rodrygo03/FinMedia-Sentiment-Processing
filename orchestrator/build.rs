fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_file = "../preprocessing/proto/preprocessing.proto";
    let proto_dir = "../preprocessing/proto";

    // Tell cargo to recompile if the proto file changes
    println!("cargo:rerun-if-changed={}", proto_file);

    tonic_build::configure()
        .build_server(true) // We need both client and server
        .build_client(true)
        .compile(&[proto_file], &[proto_dir])?;

    Ok(())
}