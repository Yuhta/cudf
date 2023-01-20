#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/io/parquet.hpp>
#include <cudf/utilities/default_stream.hpp>

int main(int argc, char** argv) {
  constexpr int64_t kDataSize = 64 << 20;
  constexpr cudf::size_type kNumCols = 8;
  constexpr int kIterations = 100;
  auto type = get_type_or_group({
      static_cast<int32_t>(data_type::INTEGRAL_SIGNED),
      static_cast<int32_t>(data_type::FLOAT),
    });
  data_profile profile;
  // profile.set_avg_run_length(256);
  auto tbl = create_random_table(
      cycle_dtypes(type, kNumCols), table_size_bytes{kDataSize}, profile);
  auto view = tbl->view();
  cudf::io::sink_info sink;
  cuio_source_sink_pair source_sink(io_type::HOST_BUFFER);
  bool write_to_file = argc > 1;
  if (write_to_file) {
    sink = cudf::io::sink_info(argv[1]);
  } else {
    sink = source_sink.make_sink_info();
  }
  cudf::io::parquet_writer_options opts = cudf::io::parquet_writer_options::builder(
      sink, view).compression(cudf::io::compression_type::NONE);
  cudf::io::write_parquet(opts);
  if (write_to_file) {
    return 0;
  }
  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(source_sink.make_source_info());
  auto t0 = std::chrono::system_clock::now();
  for (int i = 0; i < kIterations; ++i) {
    cudf::io::read_parquet(read_opts);
  }
  auto dt = std::chrono::system_clock::now() - t0;
  printf("%.2f GiB/s\n", kDataSize * kIterations * 1e9 / (1 << 30) / dt.count());
  return 0;
}
