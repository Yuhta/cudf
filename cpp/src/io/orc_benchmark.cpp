#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/io/orc.hpp>
#include <cudf/utilities/default_stream.hpp>

int main() {
  constexpr int64_t kDataSize = 128 << 20;
  constexpr cudf::size_type kNumCols = 8;
  int streamCount = 4;
  auto type = get_type_or_group({static_cast<int32_t>(data_type::INTEGRAL_SIGNED),
      static_cast<int32_t>(data_type::FLOAT),
      static_cast<int32_t>(data_type::DECIMAL),
      static_cast<int32_t>(data_type::TIMESTAMP),
      static_cast<int32_t>(data_type::STRING),
      static_cast<int32_t>(data_type::LIST),
      static_cast<int32_t>(data_type::STRUCT)});
  auto tbl = create_random_table(cycle_dtypes(type, kNumCols), table_size_bytes{kDataSize});
  auto view = tbl->view();
  cuio_source_sink_pair source_sink(io_type::HOST_BUFFER);
  cudf::io::orc_writer_options opts = cudf::io::orc_writer_options::builder(
      source_sink.make_sink_info(), view).compression(cudf::io::compression_type::ZSTD);
  cudf::io::write_orc(opts);
  cudf::io::orc_reader_options read_opts = cudf::io::orc_reader_options::builder(source_sink.make_source_info());
  std::vector<std::thread> threads;
  for (int i = 0; i < streamCount; ++i) {
    threads.emplace_back([&] { cudf::io::read_orc(read_opts); });
  }
  for (auto& t : threads) {
    t.join();
  }
  return 0;
}
