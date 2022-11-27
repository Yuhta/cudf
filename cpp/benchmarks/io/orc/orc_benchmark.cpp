#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/io/orc.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

void run(nvbench::state& state) {
  constexpr int64_t kDataSize = 64 << 20;
  auto streamCount = state.get_int64("stream_count");
  auto numCols = state.get_int64("num_cols");
  auto type = get_type_or_group({
      static_cast<int32_t>(data_type::INTEGRAL_SIGNED),
      static_cast<int32_t>(data_type::FLOAT),
      // static_cast<int32_t>(data_type::STRUCT),
      // static_cast<int32_t>(data_type::DECIMAL),
      // static_cast<int32_t>(data_type::TIMESTAMP),
      // static_cast<int32_t>(data_type::STRING),
      // static_cast<int32_t>(data_type::LIST),
    });
  data_profile profile;
  profile.set_struct_depth(1);
  std::vector<cudf::type_id> structTypes = {cudf::type_id::INT64, cudf::type_id::FLOAT32};
  profile.set_struct_types(structTypes);
  auto tbl = create_random_table(cycle_dtypes(type, numCols), table_size_bytes{kDataSize}, profile);
  auto view = tbl->view();
  cuio_source_sink_pair source_sink(io_type::HOST_BUFFER);
  cudf::io::orc_writer_options opts = cudf::io::orc_writer_options::builder(
    source_sink.make_sink_info(), view).compression(cudf::io::compression_type::ZSTD);
  cudf::io::write_orc(opts);
  cudf::io::orc_reader_options read_opts = cudf::io::orc_reader_options::builder(source_sink.make_source_info());
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer,
    [&](nvbench::launch& launch, auto& timer) {
      try_drop_l3_cache();
      timer.start();
      std::vector<std::thread> threads;
      for (int i = 0; i < streamCount; ++i) {
        threads.emplace_back([&] { cudf::io::read_orc(read_opts); });
      }
      for (auto& t : threads) {
        t.join();
      }
      timer.stop();
    });
  auto time = state.get_summary("nv/cold/time/cpu/mean").get_float64("value");
  state.add_element_count(kDataSize / time * streamCount, "bytes_per_second");
}

NVBENCH_BENCH(run).
add_int64_axis("stream_count", {1, 2, 4}).
add_int64_axis("num_cols", {16, 32});
