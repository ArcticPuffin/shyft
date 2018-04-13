#pragma once

#include <functional>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include "core/dtss_mutex.h"
#include "core/dtss_url.h"
#include "core/predictions.h"
#include "core/time_series.h"
#include "core/time_series_dd.h"
#include "time_series_info.h"
#include "utctime_utilities.h"


namespace shyft {
namespace dtss {

namespace ta = shyft::time_axis;
namespace ts = shyft::time_series;

/*
 * Data layout:
 *
 *  <krls.ts.db.file> ::
 *      "KRLS.TS.DB.0001" '\0'                  # format identifier and version, null terminated
 *      <header-start>          -> uint64_t     # number of bytes from the beginning of the file to <krls-header>
 *      <source-start>          -> uint64_t     # number of bytes from the beginning of the file to <source-url-n>
 *      <predictor-start>       -> uint64_t     # number of bytes from the beginning of the file to <predictor>
 *      <krls-header>                           # krls generic header block
 *      <source-url-n>          -> uint64_t     # length in bytes (char) of <source-url>
 *      <source-url>            -> char[n]      # source url the krls is trained on
 *      <predictor>                             # serialized predictor object
 *
 *  <krls-header> ::
 *      <scaling>               -> int64_t      # time-axis scaling
 *      <tolerance>             -> double       # krls tolerance parameter
 *      <point_fx>              -> int8_t       # point policy of predicted time-series
 *      <t_start>               -> int64_t      # earliest trained data point
 *      <t_end>                 -> int64_t      # latest trained data point
 *
 *  <predictor> ::
 *      <kernel-type-start>     -> uint64_t     # number of bytes from the beginning of the file to <kernel-type>
 *      <kernel-header-start>   -> uint64_t     # number of bytes from the beginning of the file to <kernel-header>
 *      <predictor-blob-start>  -> uint64_t     # number of bytes from the beginning of the file to <predictor-n>
 *      <kernel-type>           -> int32_t      # identifier for the kernel function
 *      <kernel-header>                         # kernel specific header, type depends on <predictor-type>
 *      <predictor-n>           -> uint64_t     # size in bytes of the following predictor blob
 *      <blob>                  -> char[n]      # serialized predictor object
 *
 *  <kernel-header> ::
 *      if <predictor-type> == krls_kernel_type_identifiers::radial_basis_kernel
 *          <rbf-gamma>         -> double       # gamma value for a radial basis function kernel
 *
 */


enum class krls_kernel_type_identifiers : std::int32_t {
    radial_basis_kernel = 1
};


struct krls_ts_db_generic_header {
    std::int64_t  scaling;  // named dt in the krls classes
    double        tolerance;
    std::int8_t   point_fx;
    std::int64_t  t_start;
    std::int64_t  t_end;

    krls_ts_db_generic_header() { }
    krls_ts_db_generic_header(std::int64_t scaling, double tolerance, ts::ts_point_fx point_fx, std::int64_t t_start, std::int64_t t_end)
        : scaling{ scaling }, tolerance{ tolerance }, point_fx{ point_fx }, t_start{ t_start }, t_end{ t_end }
    { }
};
// if this fails the the header can't be naively read and written to a file (i.e. memcopied)
static_assert(std::is_trivially_copyable_v<krls_ts_db_generic_header>,
              "\"krls_ts_db_generic_header\" needs to be a trivially copyable type");

struct krls_ts_db_rbf_header {
    double  gamma;

    krls_ts_db_rbf_header() { }
    krls_ts_db_rbf_header(double gamma)
        : gamma{ gamma }
    { }
};
// if this fails the the header can't be naively read and written to a file (i.e. memcopied)
static_assert(std::is_trivially_copyable_v<krls_ts_db_rbf_header>,
    "\"krls_ts_db_rbf_header\" needs to be a trivially copyable type");


/** \brief  Encapsulation of file io functionality.
 */
struct krls_pred_db_io {

    static constexpr std::array<char, 16> file_id{  // "KRLS.TS.DB.0001" + '\0'
        'K', 'R', 'L', 'S', '.', 'T', 'S', '.', 'D', 'B', '.', '0', '0', '0', '1', '\0'
    };

    /*  utility
     * ========= */

    static prediction::krls_rbf_predictor create_rbf_file(
        std::FILE * fh,
        const std::string & source_url,  // series filename and source url
        const core::utcperiod & period,  // period to train
        const core::utctimespan dt, const ts::ts_point_fx point_fx,
        const std::size_t dict_size, const double tolerance,  // general krls parameters
        const double gamma  // rbf kernel parameters
    ) {
        std::fseek(fh, 0, SEEK_SET);

        const krls_kernel_type_identifiers kernel_id = krls_kernel_type_identifiers::radial_basis_kernel;
        const krls_ts_db_generic_header generic_header{ dt, tolerance, point_fx, period.start, period.end };
        const krls_ts_db_rbf_header rbf_kernel_header{ gamma };
        prediction::krls_rbf_predictor predictor{};

        // file identifier
        std::fwrite(static_cast<const void*>(file_id.data()), sizeof(char), file_id.size(), fh);

        // header data
        write_header_start(fh, file_id.size()*sizeof(char) + 3*sizeof(std::uint64_t), true);
        write_header(fh, generic_header, true);

        // source url data
        write_source_url_start(fh, file_id.size()*sizeof(char) + 3*sizeof(std::uint64_t) + sizeof(krls_ts_db_generic_header), true);
        write_source_url(fh, source_url, true);

        // predictor data
        const std::uint64_t predictor_start = std::ftell(fh);
        // -----
        write_predictor_start(fh, predictor_start, true);
        // -----
        write_predictor_kernel_type_start(fh, predictor_start + 3*sizeof(std::uint64_t), true);
        write_predictor_kernel_header_start(fh, predictor_start + 3*sizeof(std::uint64_t) + sizeof(int32_t), true);
        write_predictor_blob_start(fh, predictor_start + 3*sizeof(std::uint64_t) + sizeof(int32_t) + sizeof(krls_ts_db_rbf_header), true);
        // -----
        std::fwrite(static_cast<const void *>(&kernel_id), sizeof(krls_kernel_type_identifiers), 1, fh);
        write_predictor_rbf_header(fh, rbf_kernel_header, true);
        write_predictor_rbf_predictor(fh, predictor, true);

        return predictor;
    }

    /*  pre-header data
     * ================= */

    inline static bool can_read_file(std::FILE * fh, bool skip = true) {
        if ( skip )
            std::fseek(fh, 0, SEEK_SET);

        std::remove_const_t<decltype(file_id)> data;  // ensure the type matches the header we are looking for
        std::fread(static_cast<void *>(data.data()), sizeof(char), file_id.size(), fh);

        return data == file_id;
    }

    // --------------------

    inline static void write_header_start(std::FILE * fh, const std::uint64_t start_val, bool skip = true) {
        if ( skip )
            std::fseek(fh, file_id.size()*sizeof(char), SEEK_SET);

        std::fwrite(static_cast<const void*>(&start_val), sizeof(std::uint64_t), 1, fh);
    }
    inline static std::uint64_t read_header_start(std::FILE * fh, bool skip = true) {
        if ( skip )
            std::fseek(fh, file_id.size()*sizeof(char), SEEK_SET);

        std::uint64_t skip_val;
        std::fread(static_cast<void*>(&skip_val), sizeof(std::uint64_t), 1, fh);

        return skip_val;
    }

    // --------------------

    inline static void write_source_url_start(std::FILE * fh, const std::uint64_t start_val, bool skip = true) {
        if ( skip )
            std::fseek(fh, file_id.size()*sizeof(char) + 1*sizeof(std::uint64_t), SEEK_SET);

        std::fwrite(static_cast<const void*>(&start_val), sizeof(std::uint64_t), 1, fh);
    }
    inline static std::uint64_t read_source_url_start(std::FILE * fh, bool skip = true) {
        if ( skip )
            std::fseek(fh, file_id.size()*sizeof(char) + 1*sizeof(std::uint64_t), SEEK_SET);

        std::uint64_t skip_val;
        std::fread(static_cast<void*>(&skip_val), sizeof(std::uint64_t), 1, fh);

        return skip_val;
    }

    // --------------------

    inline static void write_predictor_start(std::FILE * fh, const std::uint64_t start_val, bool skip = true) {
        if ( skip )
            std::fseek(fh, file_id.size()*sizeof(char) + 2*sizeof(std::uint64_t), SEEK_SET);

        std::fwrite(static_cast<const void*>(&start_val), sizeof(std::uint64_t), 1, fh);
    }
    inline static std::uint64_t read_predictor_start(std::FILE * fh, bool skip = true) {
        if ( skip )
            std::fseek(fh, file_id.size()*sizeof(char) + 2*sizeof(std::uint64_t), SEEK_SET);

        std::uint64_t skip_val;
        std::fread(static_cast<void*>(&skip_val), sizeof(std::uint64_t), 1, fh);

        return skip_val;
    }

    /*  header data
     * ============= */

    inline static void write_header(std::FILE * fh, const krls_ts_db_generic_header header, bool skip = true) {
        if ( skip )
            std::fseek(fh, read_header_start(fh)*sizeof(char), SEEK_SET);

        std::fwrite(static_cast<const void*>(&header), sizeof(krls_ts_db_generic_header), 1, fh);
    }
    inline static krls_ts_db_generic_header read_header(std::FILE * fh, bool skip = true) {
        if ( skip )
            std::fseek(fh, read_header_start(fh)*sizeof(char), SEEK_SET);

        krls_ts_db_generic_header header;
        std::fread(static_cast<void*>(&header), sizeof(krls_ts_db_generic_header), 1, fh);

        return header;
    }

    // --------------------

    inline static void write_source_url(std::FILE * fh, std::string url, bool skip = true) {
        if ( skip )
            std::fseek(fh, read_source_url_start(fh)*sizeof(char), SEEK_SET);

        std::uint64_t source_n = url.size();
        std::fwrite(static_cast<void*>(&source_n), sizeof(std::uint64_t), 1, fh);
        std::fwrite(static_cast<void*>(url.data()), sizeof(char), source_n, fh);
    }
    inline static std::string read_source_url(std::FILE * fh, bool skip = true) {
        if ( skip )
            std::fseek(fh, read_source_url_start(fh)*sizeof(char), SEEK_SET);

        std::uint64_t source_n;
        std::fread(static_cast<void*>(&source_n), sizeof(std::uint64_t), 1, fh);

        auto tmp_data = std::make_unique<char[]>(source_n);
        std::fread(static_cast<void*>(tmp_data.get()), sizeof(char), source_n, fh);

        return std::string{ tmp_data.get(), source_n };
    }

    /*  general predictor data
     * ======================== */

    inline static void write_predictor_kernel_type_start(std::FILE * fh, const std::uint64_t start_val, bool skip = true) {
        if ( skip )
            std::fseek(fh, read_predictor_start(fh)*sizeof(char), SEEK_SET);

        std::fwrite(static_cast<const void*>(&start_val), sizeof(std::uint64_t), 1, fh);
    }
    inline static std::uint64_t read_predictor_kernel_type_start(std::FILE * fh, bool skip = true) {
        if ( skip )
            std::fseek(fh, read_predictor_start(fh)*sizeof(char), SEEK_SET);

        std::uint64_t skip_val;
        std::fread(static_cast<void*>(&skip_val), sizeof(std::uint64_t), 1, fh);

        return skip_val;
    }

    // --------------------

    inline static void write_predictor_kernel_header_start(std::FILE * fh, const std::uint64_t start_val, bool skip = true) {
        if ( skip )
            std::fseek(fh, read_predictor_start(fh)*sizeof(char) + 1*sizeof(std::uint64_t), SEEK_SET);

        std::fwrite(static_cast<const void*>(&start_val), sizeof(std::uint64_t), 1, fh);
    }
    inline static std::uint64_t read_predictor_kernel_header_start(std::FILE * fh, bool skip = true) {
        if ( skip )
            std::fseek(fh, read_predictor_start(fh)*sizeof(char) + 1*sizeof(std::uint64_t), SEEK_SET);

        std::uint64_t skip_val;
        std::fread(static_cast<void*>(&skip_val), sizeof(std::uint64_t), 1, fh);

        return skip_val;
    }

    // --------------------

    inline static void write_predictor_blob_start(std::FILE * fh, const std::uint64_t start_val, bool skip = true) {
        if ( skip )
            std::fseek(fh, read_predictor_start(fh)*sizeof(char) + 2*sizeof(std::uint64_t), SEEK_SET);

        std::fwrite(static_cast<const void*>(&start_val), sizeof(std::uint64_t), 1, fh);
    }
    inline static std::uint64_t read_predictor_blob_start(std::FILE * fh, bool skip = true) {
        if ( skip )
            std::fseek(fh, read_predictor_start(fh)*sizeof(char) + 2*sizeof(std::uint64_t), SEEK_SET);

        std::uint64_t skip_val;
        std::fread(static_cast<void*>(&skip_val), sizeof(std::uint64_t), 1, fh);

        return skip_val;
    }

    // --------------------

    static void write_predictor_kernel_type(std::FILE * fh, const krls_kernel_type_identifiers kernel_type, bool skip = true) {
        if ( skip )
            std::fseek(fh, read_predictor_kernel_type_start(fh)*sizeof(char), SEEK_SET);

        std::fwrite(static_cast<const void*>(&kernel_type), sizeof(krls_kernel_type_identifiers), 1, fh);
    }
    static krls_kernel_type_identifiers read_predictor_kernel_type(std::FILE * fh, bool skip = true) {
        if ( skip )
            std::fseek(fh, read_predictor_kernel_type_start(fh)*sizeof(char), SEEK_SET);

        krls_kernel_type_identifiers kernel_type;
        std::fread(static_cast<void*>(&kernel_type), sizeof(krls_kernel_type_identifiers), 1, fh);

        return kernel_type;
    }

    /*  radial basis predictor data
     * ============================= */

    static void write_predictor_rbf_header(std::FILE * fh, const krls_ts_db_rbf_header kernel_type, bool skip = true) {
        if ( skip )
            std::fseek(fh, read_predictor_kernel_header_start(fh)*sizeof(char), SEEK_SET);

        std::fwrite(static_cast<const void*>(&kernel_type), sizeof(krls_ts_db_rbf_header), 1, fh);
    }
    static krls_ts_db_rbf_header read_predictor_rbf_header(std::FILE * fh, bool skip = true) {
        if ( skip )
            std::fseek(fh, read_predictor_kernel_header_start(fh)*sizeof(char), SEEK_SET);

        krls_ts_db_rbf_header kernel_header;
        std::fread(static_cast<void*>(&kernel_header), sizeof(krls_ts_db_rbf_header), 1, fh);

        return kernel_header;
    }

    // --------------------

    static void write_predictor_rbf_predictor(std::FILE * fh, const prediction::krls_rbf_predictor & predictor, bool skip = true) {
        if ( skip )
            std::fseek(fh, read_predictor_start(fh) + sizeof(krls_kernel_type_identifiers), SEEK_SET);

        std::basic_string<char> blob = predictor.to_str_blob();

        uint64_t blob_size = blob.size();
        std::fwrite(static_cast<void*>(&blob_size), sizeof(uint64_t), 1, fh);
        std::fwrite(static_cast<void*>(blob.data()), sizeof(char), blob_size, fh);
    }
    static prediction::krls_rbf_predictor read_predictor_rbf_predictor(std::FILE * fh, bool skip = true) {
        if ( skip )
            std::fseek(fh, read_predictor_start(fh) + sizeof(krls_kernel_type_identifiers), SEEK_SET);

        uint64_t blob_size;
        std::fread(static_cast<void*>(&blob_size), sizeof(uint64_t), 1, fh);

        auto blob = std::make_unique<char[]>(blob_size);
        std::fread(static_cast<void*>(blob.get()), sizeof(char), blob_size, fh);

        return prediction::krls_rbf_predictor::from_str_blob(std::basic_string<char>{ blob.get(), blob_size });
    }
};


class krls_pred_db {

public:
    using gta_t = shyft::time_axis::generic_dt;
    using gts_t = shyft::time_series::point_ts<gta_t>;
    using ts_vector_t = shyft::time_series::dd::ats_vector;
    using queries_t = std::map<std::string, std::string>;

private:
    std::string root_dir;
    std::function<ts_vector_t(const std::string &, utcperiod, bool, bool)> server_read_cb;
    // -----
    file_lock_manager f_mx;

private:
    /** helper class needed for win compensating code */
    struct close_write_handle {
        bool win_thread_close = false;
        mutable krls_pred_db * parent = nullptr;

        close_write_handle() noexcept {};// minimum fix for clang ref. https://stackoverflow.com/questions/43819314/default-member-initializer-needed-within-definition-of-enclosing-class-outside
        close_write_handle(bool wtc) noexcept : win_thread_close{ wtc } {};
        close_write_handle(const close_write_handle &) noexcept = default;

        void operator()(std::FILE * fh) const {
#ifdef _WIN32WORKAROUND
            if (win_thread_close && parent) {
                parent->fclose_me(fh);
            } else {
                std::fclose(fh); // takes forever in windows, by design
            }
#else
            std::fclose(fh);
#endif
        }
    };

    //--section dealing with windows and (postponing slow) closing files
#ifdef _WIN32WORKAROUND
    mutable mutex fclose_mx;
    mutable std::vector<std::future<void>> fclose_windows;

    void fclose_me(std::FILE *fh) {
        lock_guard<decltype(fclose_mx)> sl(fclose_mx);
        fclose_windows.emplace_back(std::async(std::launch::async, [fh]() { std::fclose(fh); }));
    }

    void wait_for_close_fh() const noexcept {
        try {
            lock_guard<decltype(fclose_mx)> scope_lock(fclose_mx);
            for (auto& fc : fclose_windows)
                fc.get();
            fclose_windows.clear();
        } catch (...) {

        }
    }
#else
    void wait_for_close_fh() const noexcept {}
#endif

public:
    krls_pred_db() = default;
    ~krls_pred_db() = default;

    /** Constructs a krls_pred_db with specified container root */
    template < typename S_CB >
    explicit krls_pred_db(const std::string& root_dir, S_CB cb)
        : root_dir{ root_dir }, server_read_cb{ cb }
    {
        if ( ! fs::is_directory(root_dir) ) {
            if ( ! fs::exists(root_dir) ) {
                if ( ! fs::create_directories(root_dir) ) {
                    throw std::runtime_error(std::string{"krls_pred_db: failed to create root directory: "} + root_dir);
                }
            } else {
                throw std::runtime_error(std::string{"krls_pred_db: designated root directory is not a directory: "} + root_dir);
            }
        }
    }

    krls_pred_db(const krls_pred_db &) = default;
    krls_pred_db(krls_pred_db &&) = default;

    krls_pred_db & operator=(const krls_pred_db &) = default;  // TODO wait for file handles to close! (windows)
    krls_pred_db & operator=(krls_pred_db &&) = default;  // TODO wait for file handles to close! (windows)

    /*  Container API
     * =============== */

public:
    void save(const std::string & fn, const gts_t & ts, bool overwrite = true, const queries_t & queries = queries_t{}, bool win_thread_close = true) {
        // // request access
        // wait_for_close_fh();
        // auto ffp = make_full_path(fn);
        // writer_file_lock lck(f_mx, ffp);

        std::string source_url;
        if ( auto it = queries.find("source_url"); it != queries.cend() ) {
            source_url = it->second;
        } else {
            throw std::runtime_error("krls_pred_db: no source url in query parameters");
        }

        core::utcperiod period = ts.total_period();

        core::utctimespan dt_scaling;
        if ( auto it = queries.find("dt_scaling"); it != queries.cend() ) {
            dt_scaling = std::stoll(it->second);
        } else {
            throw std::runtime_error("krls_pred_db: no time scaling (dt_scaling) in query parameters");
        }

        ts::ts_point_fx point_fx = ts::ts_point_fx::POINT_AVERAGE_VALUE;
        if ( auto it = queries.find("point_fx"); it != queries.cend() ) {
            if ( it->second == "average" ) {
                point_fx = ts::ts_point_fx::POINT_AVERAGE_VALUE;
            } else if ( it->second == "average" ) {
                point_fx = ts::ts_point_fx::POINT_INSTANT_VALUE;
            } else {
                throw std::runtime_error(std::string{"krls_pred_db: unknown point interpretation: "}+it->second);
            }
        }

        std::size_t krls_dict_size = 10000000u;
        if ( auto it = queries.find("krls_dict_size"); it != queries.cend() ) {
            krls_dict_size = std::stoul(it->second);
        }

        double tolerance = 0.001;
        if ( auto it = queries.find("tolerance"); it != queries.cend() ) {
            tolerance = std::stod(it->second);
        }

        double gamma = 0.001;
        if ( auto it = queries.find("gamma"); it != queries.cend() ) {
            gamma = std::stod(it->second);
        }

        this->register_rbf_series(fn, source_url, period, dt_scaling, point_fx, krls_dict_size, tolerance, gamma, win_thread_close);
    }

    gts_t read(const std::string & fn, core::utcperiod period, const queries_t & queries = queries_t{}) {
        // determine time-step
        core::utctimespan dt = core::calendar::HOUR;
        auto it = queries.find("dt");
        if ( it != queries.cend() ) {
            try {
                dt = std::stol(it->second);
            } catch ( const std::invalid_argument & ) {
                throw std::runtime_error(std::string{"krls_pred_db: cannot parse time-step: "}+it->second);
            }
        }
        // compute steps
        std::int64_t n = (period.end - period.start)/dt;
        if ( period.start + n*dt < period.end )
            n += 1;

        return this->predict_time_series(fn, gta_t{ period.start, dt, static_cast<std::size_t>(n) });
    }

    void remove(const std::string & fn, const queries_t & queries = queries_t{}) {
        wait_for_close_fh();
        auto ffp = make_full_path(fn);
        writer_file_lock lck(f_mx, ffp);
        // TODO
    }

    ts_info get_ts_info(const std::string & fn, const queries_t & queries = queries_t{}) {
        wait_for_close_fh();
        auto ffp = make_full_path(fn);
        reader_file_lock lck(f_mx, ffp);
        // TODO
        return ts_info{};
    }

    std::vector<ts_info> find(const std::string & match, const queries_t & queries = queries_t{}) {
        wait_for_close_fh();
        // TODO
        return std::vector<ts_info>{};
    };

    /*  KRLS container API
     * ==================== */

public:
    void register_rbf_series(
        const std::string & fn, const std::string & source_url,  // series filename and source url
        const core::utcperiod & period,  // period to train
        const core::utctimespan dt, const ts::ts_point_fx point_fx,
        const std::size_t dict_size, const double tolerance,  // general parameters
        const double gamma,  // rbf kernel parameters
        const bool win_thread_close = true
    ) {
        // request exclusive access
        wait_for_close_fh();
        auto ffp = make_full_path(fn);
        writer_file_lock lck(f_mx, ffp);
        // setup file pointer
        std::unique_ptr<std::FILE, close_write_handle> fh;  // zero-initializes deleter
        fh.get_deleter().win_thread_close = win_thread_close;
        fh.get_deleter().parent = const_cast<krls_pred_db*>(this);

        if ( ! save_path_exists(fn) ) {
            // open for binary read/write, destroy any contents
            fh.reset(std::fopen(ffp.c_str(), "w+b"));
            // setup datafile and create predictor
            prediction::krls_rbf_predictor predictor = krls_pred_db_io::create_rbf_file(fh.get(),
                source_url, period, dt, point_fx, dict_size, tolerance, gamma);
            // train on data available using the source url
            this->train_on_period(predictor, period, source_url);
            // write the trained predictor
            krls_pred_db_io::write_predictor_rbf_predictor(fh.get(), predictor);
        } else {
            throw std::runtime_error(std::string{"krls_pred_db: series already registered: "} + fn);
        }
    }

    gts_t predict_time_series(
        const std::string & fn,
        const gta_t & ta
    ) {
        // request shared access
        wait_for_close_fh();
        auto ffp = make_full_path(fn);
        reader_file_lock lck(f_mx, ffp);

        if ( save_path_exists(fn) ) {
            // setup file pointer
            std::unique_ptr<std::FILE, decltype(&std::fclose)> fh{ std::fopen(ffp.c_str(), "rb"), &std::fclose };

            if ( krls_pred_db_io::can_read_file(fh.get()) ) {
                const krls_kernel_type_identifiers kernel_type = krls_pred_db_io::read_predictor_kernel_type(fh.get());
                switch ( kernel_type ) {

                case krls_kernel_type_identifiers::radial_basis_kernel: {
                    prediction::krls_rbf_predictor predictor = krls_pred_db_io::read_predictor_rbf_predictor(fh.get());
                    return predictor.predict<gts_t, gta_t>(ta);
                }

                default:
                    throw std::runtime_error(std::string{"krls_pred_db: unknown kernel identifier: "} + std::to_string(static_cast<std::int32_t>(kernel_type)));
                }
            } else {
                throw std::runtime_error(std::string{"krls_pred_db: cannot read, unknown data format: "}+fn);
            }
        } else {
            throw std::runtime_error(std::string{"krls_pred_db: no data for id: "}+fn);
        }
    }

    /*  Internal implementation
     * ========================= */

private:
    bool save_path_exists(const std::string & fn) const {
        fs::path fn_path{ fn }, root_path{ root_dir };
        if ( fn_path.is_relative() ) {
            fn_path = root_path / fn_path;
        } else {  /* fn_path.is_absolute() */
            // questionable: should we allow outside container specs?
            //  - if determined determined to be allowed: remove this branch
            throw std::runtime_error("krls_pred_db: outside container spec not allowed");
        }
        return fs::is_regular_file(fn_path);
    }
    std::string make_full_path(const std::string& fn, bool create_paths = false) const {
        fs::path fn_path{ fn }, root_path{ root_dir };
        // determine path type
        if (fn_path.is_relative()) {
            fn_path = root_path / fn_path;
        } else {  /* fn_path.is_absolute() */
            // questionable: should we allow outside container specs?
            //  - if determined determined to be allowed: remove this branch
            throw std::runtime_error("krls_pred_db: outside container spec not allowed");
        }
        // not a directory and create missing path
        if (fs::is_directory(fn_path)) {
            throw std::runtime_error(std::string{"krls_pred_db: "}+fn_path.string()+" is a directory. Should be a file.");
        } else if (!fs::exists(fn_path) && create_paths) {
            fs::path rp = fn_path.parent_path();
            if (rp.compare(root_path) > 0) {  // if fn contains sub-directory, we have to check that it exits
                if (!fs::is_directory(rp)) {
                    fs::create_directories(rp);
                }
            }
        }
        // -----
        return fn_path.string();
    }

    // --------------------

    void train_on_period(
        prediction::krls_rbf_predictor & predictor,
        const core::utcperiod & period,
        const std::string & source_url
    ) {
        ts_vector_t vec = this->server_read_cb(source_url, period, false, false);

        if ( vec.size() > 0 ) {
            for ( auto && ts : vec ) {
                predictor.train(ts);
            }
        } else {
            throw std::runtime_error(std::string{"krls_pred_db: no time-series at url: "} + source_url);
        }
    }

};

}
}
