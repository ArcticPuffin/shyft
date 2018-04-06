#pragma once

#include <functional>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include "core/predictions.h"
#include "core/time_series_dd.h"
#include "time_series_info.h"
#include "utctime_utilities.h"


namespace shyft {
namespace dtss {

/*
 * Data layout:
 *
 *  <krls.ts.db.file> ::
 *      "KRLS.TS.DB.0001" '\0'              # format identifier and version, null terminated
 *      <header-skip>     -> uint64_t       # number of bytes from the beginning of the file to the header block
 *      <predictor-skip>  -> uint64_t       # number of bytes from the beginning of the file to the predictor block
 *      <header>                            # header block
 *      <predictor>                         # serialized predictor object
 *  
 *  <header> ::
 *      <scaling>         -> int64_t        # time-axis scaling
 *      <tolerance>       -> double         # krls tolerance parameter
 *      <t_start>         -> int64_t        # earliest trained data point
 *      <t_end>           -> int64_t        # latest trained data point
 *
 *  <predictor> ::
 *      <kernel-type>     -> int32_t        # identifier for the kernel function
 *      <predictor-data>
 *
 *  <predictor-data> ::
 *      if <kernel-type> == krls_kernel_type_identifiers::radial_basis_kernel
 *          <predictor-n> -> uint64_t       # size in bytes of the following predictor blob
 *          <blob>                          # serialized predictor object
 */


enum class krls_kernel_type_identifiers : std::int32_t {
    radial_basis_kernel = 1
};


struct krls_ts_db_header {
    std::int64_t  scaling;
    double        tolerance;
    std::int64_t  t_start;
    std::int64_t  t_end;

    krls_ts_db_header() { }
    krls_ts_db_header(std::int64_t scaling, double tolerance, std::int64_t t_start, std::int64_t t_end)
        : scaling{ scaling }, tolerance{ tolerance }, t_start{ t_start }, t_end{ t_end }
    { }
};
// if this fails the the header can't be naively read and written to a file (i.e. memcopied)
static_assert(std::is_trivially_copyable_v<krls_ts_db_header>,
              "\"krls_ts_db_header\" needs to be a trivially copyable type");


/** \brief  Encapsulation of file io functionality.
 */
struct krls_pred_db_io {

    static constexpr std::array<char, 16> file_id{  // "KRLS.TS.DB.0001" + '\0'
        'K', 'R', 'L', 'S', '.', 'T', 'S', '.', 'D', 'B', '.', '0', '0', '0', '1', '\0'
    };

    /*  pre-header data
     * ================= */

    static bool can_read_file(std::FILE * fh) {
        std::fseek(fh, 0, SEEK_SET);

        std::remove_const_t<decltype(file_id)> data;  // ensure the type matches the header we are looking for
        std::fread(static_cast<void *>(data.data()), sizeof(char), file_id.size(), fh);

        return data == file_id;
    }

    // --------------------

    static std::uint64_t read_header_start(std::FILE * fh) {
        std::fseek(fh, file_id.size()*sizeof(char), SEEK_SET);

        std::uint64_t skip_val;
        std::fread(static_cast<void*>(&skip_val), sizeof(std::uint64_t), 1, fh);

        return skip_val;
    }

    static std::uint64_t read_predictor_start(std::FILE * fh) {
        std::fseek(fh, file_id.size()*sizeof(char) + sizeof(std::uint64_t), SEEK_SET);

        std::uint64_t skip_val;
        std::fread(static_cast<void*>(&skip_val), sizeof(std::uint64_t), 1, fh);

        return skip_val;
    }

    /*  header data
     * ============= */

    static krls_ts_db_header read_header(std::FILE * fh) {
        std::fseek(fh, read_header_start(fh), SEEK_SET);

        krls_ts_db_header header;
        std::fread(static_cast<void*>(&header), sizeof(krls_ts_db_header), 1, fh);

        return header;
    }

    /*  predictor data
     * ================ */

    static krls_kernel_type_identifiers read_predictor_kernel_type(std::FILE * fh) {
        std::fseek(fh, read_predictor_start(fh), SEEK_SET);

        krls_kernel_type_identifiers kernel_type;
        std::fread(static_cast<void*>(&kernel_type), sizeof(krls_kernel_type_identifiers), 1, fh);

        return kernel_type;
    }

    static double read_predictor_rbf_gamma(std::FILE * fh) {
        std::fseek(fh, read_predictor_start(fh) + sizeof(krls_kernel_type_identifiers), SEEK_SET);

        double gamma_val;
        std::fread(static_cast<void*>(&gamma_val), sizeof(double), 1, fh);

        return gamma_val;
    }

    static void write_predictor_rbf_predictor(std::FILE * fh, const prediction::krls_rbf_predictor & predictor) {
        std::fseek(fh, read_predictor_start(fh) + sizeof(krls_kernel_type_identifiers), SEEK_SET);

        std::string blob = predictor.to_str_blob();

        uint64_t blob_size = blob.size();
        std::fwrite(static_cast<void*>(&blob_size), sizeof(uint64_t), 1, fh);
        std::fwrite(static_cast<void*>(blob.data()), sizeof(char), blob_size, fh);

        // TODO Figure out what serialization method to use. Especially size.
    }

    static prediction::krls_rbf_predictor read_predictor_rbf_predictor(std::FILE * fh) {
        std::fseek(fh, read_predictor_start(fh) + sizeof(krls_kernel_type_identifiers) + sizeof(double), SEEK_SET);

        uint64_t blob_size;
        std::fread(static_cast<void*>(&blob_size), sizeof(uint64_t), 1, fh);

        std::unique_ptr<char[]> blob = std::make_unique<char[]>(blob_size);
        std::fread(static_cast<void*>(blob.get()), sizeof(char), blob_size, fh);

        return prediction::krls_rbf_predictor::from_str_blob(std::string{ blob.get(), blob_size });
    }
};


class krls_pred_db {

    using gta_t = shyft::time_axis::generic_dt;
    using gts_t = shyft::time_series::point_ts<gta_t>;
    using ts_vector_t = shyft::time_series::dd::ats_vector;
    using queries_t = std::map<std::string, std::string>;

    std::string root_dir;
    std::function<ts_vector_t(const std::string &, utcperiod, bool, bool)> server_read_cb;

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
            }
        } else {
            throw std::runtime_error(std::string{"krls_pred_db: designated root directory is not a directory: "} + root_dir);
        }
    }

    krls_pred_db(const krls_pred_db &) = default;
    krls_pred_db(krls_pred_db &&) = default;

    krls_pred_db & operator=(const krls_pred_db &) = default;  // TODO wait for file handles to close! (windows)
    krls_pred_db & operator=(krls_pred_db &&) = default;  // TODO wait for file handles to close! (windows)

    /*  Container API
     * =============== */

public:
    void save(const std::string & fn, const gts_t & ts, bool overwrite = true, const queries_t & queries = queries_t{}, bool win_thread_close = true) const {
        // only ts_id -> lookup from server
        // ts_id and data -> train on data first, then check server for more
    }

    gts_t read(const std::string & fn, core::utcperiod p, const queries_t & queries = queries_t{}) const {
        // TODO
        return gts_t{};
    }

    void remove(const std::string & fn, const queries_t & queries = queries_t{}) const {
        // TODO
    }

    ts_info get_ts_info(const std::string & fn, const queries_t & queries = queries_t{}) const {
        // TODO
        return ts_info{};
    }

    std::vector<ts_info> find(const std::string & match, const queries_t & queries = queries_t{}) const {
        // TODO
        return std::vector<ts_info>{};
    };


    /*  Internal implementation
     * ========================= */

// private:


};

}
}
