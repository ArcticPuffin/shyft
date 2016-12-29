#include "boostpython_pch.h"
#include <boost/python/docstring_options.hpp>

//-- for serialization:
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

//-- notice that boost serialization require us to
//   include shared_ptr/vector .. etc.. wherever it's needed

#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/vector.hpp>

//-- for the server
#include <dlib/server.h>
#include <dlib/iosockstream.h>

//#include "core/timeseries.h"
#include "api/timeseries.h"

// from https://www.codevate.com/blog/7-concurrency-with-embedded-python-in-a-multi-threaded-c-application
namespace boost {
    namespace python {

        struct release_gil_policy {
            // Ownership of this argument tuple will ultimately be adopted by
            // the caller.
            template <class ArgumentPackage>
            static bool precall(ArgumentPackage const&) {
                // Release GIL and save PyThreadState for this thread here

                return true;
            }

            // Pass the result through
            template <class ArgumentPackage>
            static PyObject* postcall(ArgumentPackage const&, PyObject* result) {
                // Reacquire GIL using PyThreadState for this thread here

                return result;
            }

            typedef default_result_converter result_converter;
            typedef PyObject* argument_package;

            template <class Sig>
            struct extract_return_type : mpl::front<Sig> {
            };

        private:
            // Retain pointer to PyThreadState on a per-thread basis here

        };
    }
}

struct scoped_gil_release {
    scoped_gil_release() noexcept {
        py_thread_state = PyEval_SaveThread();
    }
    ~scoped_gil_release() noexcept {
        PyEval_RestoreThread(py_thread_state);
    }
private:
    PyThreadState * py_thread_state;
};

struct scoped_gil_aquire {
    scoped_gil_aquire() noexcept {
        py_state = PyGILState_Ensure();
    }
    ~scoped_gil_aquire() noexcept {
        PyGILState_Release(py_state);
    }
private:
    PyGILState_STATE   py_state;
};

namespace shyft {
    namespace dtss {

        template<class T>
        static api::apoint_ts read_ts(T& in) {
            int sz;
            in.read((char*)&sz, sizeof(sz));
            std::vector<char> blob(sz, 0);
            in.read((char*)blob.data(), sz);
            return api::apoint_ts::deserialize_from_bytes(blob);
        }

        template <class T>
        static void  write_ts(const api::apoint_ts& ats, T& out) {
            auto blob = ats.serialize_to_bytes();
            int sz = blob.size();
            out.write((const char*)&sz, sizeof(sz));
            out.write((const char*)blob.data(), sz);
        }

        template <class TSV,class T>
        static void write_ts_vector(TSV &&ats, T & out) {
            int sz = ats.size();
            out.write((const char*)&sz, sizeof(sz));
            for (const auto & ts : ats)
                write_ts(ts, out);
        }

        template<class T>
        static std::vector<api::apoint_ts> read_ts_vector(T& in) {
            int sz;
            in.read((char*)&sz, sizeof(sz));
            std::vector<api::apoint_ts> r;
            r.reserve(sz);
            for (int i = 0;i < sz;++i)
                r.push_back(read_ts(in));
            return r;
        }
        enum dtss_message {
            EVALUATE_TS_VECTOR,
            //CACHE
            //FIND
            //etc. etc.
        };



        struct dtss_server : dlib::server_iostream {
            boost::python::object cb;
            dtss_server() {
                PyEval_InitThreads();// ensure threads-is enabled
            }
            template<class TSV>
            std::vector<api::apoint_ts> do_evaluate_ts_vector(core::utcperiod bind_period, TSV&& atsv) {
                //-- just for the testing create dummy-ts here.
                // later: collect all bind_info.ref, collect by match into list, then invoke handler,
                //        then invoke default handler for the remaining not matching a bind-handlers (like a regexpr ?)
                //
                core::calendar utc;
                time_axis::generic_dt ta(bind_period.start, core::deltahours(1), bind_period.timespan()/api::deltahours(1));
                api::apoint_ts dummy_ts(ta, 1.0, timeseries::POINT_AVERAGE_VALUE);
                std::map<std::string,api::ts_bind_info> ts_bind_map;
                std::vector<std::string> ts_id_list;
                for (auto& ats : atsv) {
                    auto ts_refs = ats.find_ts_bind_info();
                    for(const auto& bi:ts_refs) {
                        if (ts_bind_map.find(bi.reference) == ts_bind_map.end()) { // maintain unique set
                            ts_id_list.push_back(bi.reference);
                            ts_bind_map[bi.reference] = bi;
                        }
                    }
                }
                auto bts=fire_cb(ts_id_list,bind_period);
                if(bts.size()!=ts_id_list.size())
                    throw std::runtime_error(std::string("failed to bind all of ")+std::to_string(bts.size())+std::string(" ts"));
                for(size_t i=0;i<ts_id_list.size();++i) {
                    try {
                        //std::cout<<"bind "<<i<<": "<<ts_id_list[i]<<":"<<bts[i].size()<<"\n";
                        ts_bind_map[ts_id_list[i]].ts.bind(bts[i]);
                    } catch(const std::runtime_error&re) {
                        std::cout<<"failed to bind "<<ts_id_list[i]<<re.what()<<"\n";
                    }
                }
                //-- evaluate, when all binding is done (vectorized calc.
                std::vector<api::apoint_ts> evaluated_tsv;
                int i=0;
                for (auto &ats : atsv) {
                    try {
                        evaluated_tsv.emplace_back(ats.time_axis(), ats.values(), ats.point_interpretation());
                    } catch(const std::runtime_error&re) {
                        std::cout<<"failed to evalutate ts:"<<i<<"::"<<re.what()<<"\n";
                    }
                    i++;
                }
                return evaluated_tsv;
            }

            static int msg_count ;

            std::vector<api::apoint_ts> fire_cb(std::vector<std::string>ts_ids,core::utcperiod p) {
                std::vector<api::apoint_ts> r;
                if (cb.ptr()!=Py_None) {
                    scoped_gil_aquire gil;
                    r = boost::python::call<std::vector<api::apoint_ts>>(cb.ptr(), ts_ids, p);
                }
                return r;
            }
            void process_messages(int msec) {
                scoped_gil_release gil;
                //start_async();
                std::this_thread::sleep_for(std::chrono::milliseconds(msec));
            }

            void on_connect(
                std::istream& in,
                std::ostream& out,
                const std::string& foreign_ip,
                const std::string& local_ip,
                unsigned short foreign_port,
                unsigned short local_port,
                dlib::uint64 connection_id
            ) {
                while (in.peek() != EOF) {
                    int msg_type;
                    in.read((char*)&msg_type, sizeof(msg_type));
                    msg_count++;
                    switch ((dtss_message)msg_type) {
                        case EVALUATE_TS_VECTOR: {
                            core::utcperiod bind_period;
                            in.read((char*)&bind_period, sizeof(bind_period));
                            write_ts_vector(do_evaluate_ts_vector(bind_period, read_ts_vector(in)),out);
                        } break;
                        default:
                            throw std::runtime_error(std::string("Got unknown message type:") + std::to_string(msg_type));
                    }
                }
            }
        };
        int dtss::dtss_server::msg_count = 0;
        std::vector<api::apoint_ts> dtss_evaluate(std::string host_port,const std::vector<api::apoint_ts>& tsv, core::utcperiod p) {
            scoped_gil_release gil;
            dlib::iosockstream io(host_port);
            int msg_type = EVALUATE_TS_VECTOR;
            io.write((const char*) &msg_type, sizeof(msg_type));
            io.write((const char*)&p, sizeof(p));
            write_ts_vector(tsv, io);
            return read_ts_vector(io);
        }
    }
}
namespace expose {
    //using namespace shyft::core;
    using namespace boost::python;

    static void dtss_messages() {

    }
    static void dtss_server() {
        typedef shyft::dtss::dtss_server DtsServer;
        //bases<>,std::shared_ptr<DtsServer>
        class_<DtsServer, boost::noncopyable >("DtsServer")
            .def("set_listening_port", &DtsServer::set_listening_port, args("port_no"), "tbd")
            .def("start_async",&DtsServer::start_async)
            .def("set_max_connections",&DtsServer::set_max_connections,args("max_connect"),"tbd")
            .def("get_max_connections",&DtsServer::get_max_connections,"tbd")
            .def("clear",&DtsServer::clear,"stop serving connections")
            .def("is_running",&DtsServer::is_running,"true if server is listening and running")
            .def("get_listening_port",&DtsServer::get_listening_port,"returns the port number it's listening at")
            .def_readwrite("cb",&DtsServer::cb,"callback for binding")
            .def("fire_cb",&DtsServer::fire_cb,args("msg","rp"),"testing fire from c++")
            .def("process_messages",&DtsServer::process_messages,args("msec"),"wait and process messages for specified number of msec before returning")
            //.add_static_property("msg_count",
            //                     make_getter(&DtsServer::msg_count),
            //                     make_setter(&DtsServer::msg_count),"total number of requests")
            ;

    }
    static void dtss_client() {
        def("dtss_evaluate", shyft::dtss::dtss_evaluate, args("host_port","ts_vector","utcperiod"),
            "tbd"
            );

    }

    void dtss() {
        dtss_messages();
        dtss_server();
        dtss_client();
    }
}
