﻿#include "test_pch.h"

#include "core/dtss.h"
#include "core/dtss_cache.h"
#include "core/dtss_client.h"

#include "core/utctime_utilities.h"
#include "core/time_axis.h"
#include "core/time_series_merge.h"
#include "core/time_series_dd.h"


#include <future>
#include <mutex>
#include <regex>
#include <boost/filesystem.hpp>
#include <cstdint>
#ifdef _WIN32
#include <io.h>
#else
#include <sys/io.h>
#define O_BINARY 0
#define O_SEQUENTIAL 0
#include <sys/stat.h>
#endif
#include <fcntl.h>
#include <string_view>

namespace  fs=boost::filesystem;
#include <armadillo>

using namespace std;
using namespace shyft;
using namespace shyft::core;
using shyft::time_series::dd::apoint_ts;
using shyft::time_series::dd::gta_t;

apoint_ts mk_expression(utctime t, utctimespan dt, int n) {

    std::vector<double> x; x.reserve(n);
    for (int i = 0; i < n; ++i)
        x.push_back(-double(n) / 2.0 + i);
    apoint_ts aa(gta_t(t, dt, n), x);
    auto a = aa*3.0 + aa;
    return a;
}

dlib::logger dlog("dlib.log");

#define TEST_SECTION(x)

TEST_SUITE("dtss") {

TEST_CASE("dtss_lru_cache") {
    using shyft::dtss::lru_cache;
    using std::map;
    using std::list;
    using std::vector;
    using std::string;
    using std::back_inserter;
    using shyft::time_series::dd::apoint_ts;
    using shyft::time_series::dd::gta_t;
    const auto stair_case=shyft::time_series::POINT_AVERAGE_VALUE;
    lru_cache<string, apoint_ts, map > c(2);

    apoint_ts r;
    vector<string> mru;
    gta_t ta(0, 1, 10);

    TEST_SECTION("empty_cache") {
        FAST_CHECK_UNARY_FALSE(c.try_get_item("a", r));
    }
    TEST_SECTION("add_one_item") {
        c.add_item("a", apoint_ts(ta, 1.0, stair_case));
        FAST_CHECK_UNARY(c.try_get_item("a", r));
        FAST_CHECK_EQ(ta.size(), r.time_axis().size());
        FAST_CHECK_UNARY_FALSE(c.try_get_item("b", r));
    }
    TEST_SECTION("add_second_item") {
        c.add_item("b", apoint_ts(ta, 2.0, stair_case));
        FAST_CHECK_UNARY(c.try_get_item("a", r));
        FAST_CHECK_UNARY(c.try_get_item("b", r));
        c.get_mru_keys(back_inserter(mru));
        FAST_CHECK_EQ(string("b"), mru[0]);
        FAST_CHECK_EQ(string("a"), mru[1]);
    }
    TEST_SECTION("mru_item_in_front") {
        c.try_get_item("a", r);
        mru.clear(); c.get_mru_keys(back_inserter(mru));
        FAST_CHECK_EQ(string("a"), mru[0]);
        FAST_CHECK_EQ(string("b"), mru[1]);
    }
    TEST_SECTION("excessive_lru_item_evicted_when_adding") {
        c.add_item("c", apoint_ts(ta, 3.0, stair_case));
        FAST_CHECK_UNARY_FALSE(c.try_get_item("b", r));
        FAST_CHECK_UNARY(c.try_get_item("c", r));
        FAST_CHECK_UNARY(c.try_get_item("a", r));
    }
    TEST_SECTION("remove_item") {
        c.remove_item("a");
        FAST_CHECK_UNARY_FALSE(c.try_get_item("a", r));
    }
    TEST_SECTION("ensure_items_added_are_first") {
        c.add_item("d", apoint_ts(ta, 4.0, stair_case));
        mru.clear(); c.get_mru_keys(back_inserter(mru));
        FAST_CHECK_EQ(string("d"), mru[0]);
        FAST_CHECK_EQ(string("c"), mru[1]);
    }
    TEST_SECTION("update_existing") {
        c.try_get_item("c", r);//just to ensure "c" is in first position
        c.add_item("d", apoint_ts(ta, 4.2, stair_case)); //update "d"
        c.try_get_item("d", r);
        FAST_CHECK_GT(r.value(0), 4.1);
        mru.clear(); c.get_mru_keys(back_inserter(mru));
        FAST_CHECK_EQ(string("d"), mru[0]);
        FAST_CHECK_EQ(string("c"), mru[1]);
    }
}
TEST_CASE("dtss_ts_cache") {
    using std::vector;
    using std::string;
    using shyft::core::utctime;
    using shyft::core::deltahours;
    using shyft::dtss::cache_stats;
    using shyft::time_series::dd::apoint_ts;
    using shyft::time_series::dd::gta_t;
    using shyft::dtss::apoint_ts_frag;
    using dtss_cache=shyft::dtss::cache<apoint_ts_frag,apoint_ts>;
    const auto stair_case=shyft::time_series::POINT_AVERAGE_VALUE;
    size_t max_ids=10;

    dtss_cache c(max_ids);

    apoint_ts x;
    utcperiod p{0,10};
    FAST_CHECK_EQ(false,c.try_get("a",p,x));

    utctime t0{5};
    utctime t1{10};

    utctimespan dt{1};
    size_t n{3};
    apoint_ts ts_a{gta_t{t0,dt,n},1.0,stair_case};
    apoint_ts ts_a2{gta_t{t1,dt,n},1.0,stair_case};

    c.add("a",ts_a);
    FAST_REQUIRE_EQ(true,c.try_get("a",utcperiod{t0,t0+dt},x));
    FAST_CHECK_EQ(1.0,x.value(0));

    c.add("a",ts_a2);// test we can add twice, a fragment (replace)

    FAST_REQUIRE_EQ(false,c.try_get("a",utcperiod{t0,t0+ 10*dt},x));

    auto s=c.get_cache_stats();
    FAST_CHECK_EQ(s.hits,2);
    FAST_CHECK_EQ(s.misses,1);
    FAST_CHECK_EQ(s.coverage_misses,1);
    FAST_CHECK_EQ(s.point_count,6);
    FAST_CHECK_EQ(s.fragment_count,2);
    FAST_CHECK_EQ(s.id_count,1);

    c.clear_cache_stats();
    c.flush();
    s = c.get_cache_stats();
    FAST_CHECK_EQ(s.hits, 0);
    FAST_CHECK_EQ(s.misses, 0);
    FAST_CHECK_EQ(s.coverage_misses, 0);
    FAST_CHECK_EQ(s.point_count, 0);
    FAST_CHECK_EQ(s.fragment_count, 0);
    FAST_CHECK_EQ(s.id_count, 0);

    c.add("a", ts_a2);

    c.remove("a");
    FAST_REQUIRE_EQ(false,c.try_get("a",utcperiod{t0,t0+dt},x));

    //-- test vector operations
    // arrange n_ts
    vector<string> ids;
    vector<apoint_ts> tss;
    size_t n_ts = 3;
    gta_t mta{ t0,dt,n };
    for (size_t i = 0; i<n_ts; ++i) {
        ids.push_back(to_string(i));
        tss.emplace_back(mta, double(i), stair_case);
    }

    c.add(ids, tss); // add a vector of ids|tss

    auto mts = c.get(ids, mta.total_period());// get a vector of  ids back as map[id]=ts
    FAST_REQUIRE_EQ(n_ts, mts.size());
    for (size_t i = 0; i<n_ts; ++i) {
        FAST_REQUIRE_UNARY(mts.find(ids[i])!=mts.end());
        FAST_CHECK_EQ(mts[ids[i]].value(0), double(i)); // just check one value unique for ts.
    }

    auto ids2 = ids; ids2.push_back("not there");// ask for something that's not there
    auto mts2 = c.get(ids2, mta.total_period());
    FAST_REQUIRE_EQ(n_ts, mts.size());
    for (size_t i = 0; i<n_ts; ++i) {
        FAST_REQUIRE_UNARY(mts2.find(ids[i]) != mts2.end());
        FAST_CHECK_EQ(mts2[ids[i]].value(0), double(i)); // just check one value unique for ts.
    }

    c.remove(ids2); // remove by vector (even with elem not there)
    s = c.get_cache_stats();
    FAST_CHECK_EQ(s.point_count, 0);
    FAST_CHECK_EQ(s.fragment_count, 0);
    FAST_CHECK_EQ(s.id_count, 0);

}
TEST_CASE("dtss_mini_frag") {
    using std::vector;
    using std::string;
    using std::min;
    using std::max;
    using shyft::core::utcperiod;
    using shyft::core::utctime;
    using shyft::dtss::mini_frag;
    using shyft::time_axis::continuous_merge;

    struct tst_frag {
        utcperiod p;
        int id{0};
        tst_frag(utctime f,utctime u):p(f,u){}
        tst_frag(utctime f,utctime u,int id):p(f,u),id(id){}
        utcperiod total_period() const {return p;}
        size_t size() const {return size_t(p.timespan());}
        tst_frag merge(const tst_frag&o) const {
            if(!continuous_merge(p,o.p))
                throw runtime_error("Wrong merge op attempted");
            return tst_frag{min(o.p.start,p.start),max(o.p.end,p.end)};
        }
    };

    mini_frag<tst_frag> m;

    FAST_CHECK_EQ(m.count_fragments(),0);
    FAST_CHECK_EQ(m.estimate_size(),0);
    FAST_CHECK_EQ(m.get_ix(utcperiod(0,10)),string::npos);

    m.add(tst_frag(5,10));// add first
    FAST_CHECK_EQ(m.count_fragments(),1);
    FAST_CHECK_EQ(m.estimate_size(),5);
    FAST_CHECK_EQ(m.get_ix(utcperiod{5,10}),0);

    m.add(tst_frag(4,5));// add merge element
    FAST_CHECK_EQ(m.count_fragments(),1);
    FAST_CHECK_EQ(m.estimate_size(),6);
    FAST_CHECK_EQ(m.get_ix(utcperiod{4,6}),0);

    m.add(tst_frag(1,3));// add before first
    FAST_CHECK_EQ(m.count_fragments(),2);
    FAST_CHECK_EQ(m.estimate_size(),8);
    FAST_CHECK_EQ(m.get_ix(utcperiod{4,6}),1);
    FAST_CHECK_EQ(m.get_ix(utcperiod{1,2}),0);

    m.add(tst_frag(3,4));// add a piece that merge [0] [1]
    FAST_CHECK_EQ(m.count_fragments(),1);
    FAST_CHECK_EQ(m.estimate_size(),9);
    FAST_CHECK_EQ(m.get_ix(utcperiod{4,6}),0);

    m.add(tst_frag{11,12}); // append a frag at the end
    FAST_CHECK_EQ(m.count_fragments(),2);
    FAST_CHECK_EQ(m.estimate_size(),10);
    FAST_CHECK_EQ(m.get_ix(utcperiod{4,6}),0);
    FAST_CHECK_EQ(m.get_ix(utcperiod{11,12}),1);
    FAST_CHECK_EQ(m.get_ix(utcperiod{11,13}),string::npos);
    FAST_CHECK_EQ(m.get_ix(utcperiod{1,12}),string::npos);

    m.add(tst_frag{13,15}); // append a frag at the end
    FAST_CHECK_EQ(m.count_fragments(),3);
    FAST_CHECK_EQ(m.estimate_size(),12);
    FAST_CHECK_EQ(m.get_ix(utcperiod{13,14}),2);
    FAST_CHECK_EQ(m.get_ix(utcperiod{11,12}),1);
    FAST_CHECK_EQ(m.get_ix(utcperiod{5,6}),0);

    m.add(tst_frag{11,14});// append a frag that melts [1]..[2] into one
    FAST_CHECK_EQ(m.count_fragments(),2);
    FAST_CHECK_EQ(m.get_ix(utcperiod{11,15}),1);

    m.add(tst_frag{0,20}); // append a frag that melts all into one
    FAST_CHECK_EQ(m.count_fragments(),1);
    FAST_CHECK_EQ(m.estimate_size(),20);

    m.add(tst_frag{0,20,1}); // append a marked frag that is exactly equal
    FAST_CHECK_EQ(m.count_fragments(),1);
    FAST_CHECK_EQ(m.estimate_size(),20);
    FAST_CHECK_EQ(m.get_ix(utcperiod{0,10}),0);
    FAST_CHECK_EQ(m.get_by_ix(0).id,1);// verify we got the marked in place

    m.add(tst_frag{21,23});// two frags
    m.add(tst_frag{25,27});// three
    m.add(tst_frag{0,23});// exactly cover second frag
    FAST_CHECK_EQ(m.count_fragments(),2);

    m.add(tst_frag{1,24});
    FAST_CHECK_EQ(m.count_fragments(),2);

    m.add(tst_frag{2,27});// parts p1, p2.end
    FAST_CHECK_EQ(m.count_fragments(),1);

    m.add(tst_frag{-1,27});
    FAST_CHECK_EQ(m.count_fragments(),1);

}

TEST_CASE("dlib_server_basics") {
    dlog.set_level(dlib::LALL);
    dlib::set_all_logging_output_streams(std::cout);
    dlib::set_all_logging_levels(dlib::LALL);
    dlog << dlib::LINFO << "Starting dtss test";
    using namespace shyft::dtss;
    try {
        // Tell the server to begin accepting connections.
        calendar utc;
        auto t = utc.time(2016, 1, 1);
        auto dt = deltahours(1);
        auto dt24 = deltahours(24);
        int n = 240;
        int n24 = n / 24;
        time_axis::fixed_dt ta(t, dt, n);
        gta_t ta24(t, dt24, n24);
        bool throw_exception = false;
        read_call_back_t cb = [ta, &throw_exception](id_vector_t ts_ids, core::utcperiod p)
            ->ts_vector_t {
            ts_vector_t r; r.reserve(ts_ids.size());
            double fv = 1.0;
            for (size_t i = 0; i < ts_ids.size(); ++i)
                r.emplace_back(ta, fv += 1.0);
            if (throw_exception) {
                dlog << dlib::LINFO << "Throw from inside dtss executes!";
                throw std::runtime_error("test exception");
            }
            return r;
        };
        std::vector<std::string> ts_names = {
            string("a.prod.mw"),
            string("b.prod.mw")
        };
        find_call_back_t fcb = [&ts_names](std::string search_expression)
            ->ts_info_vector_t {
            ts_info_vector_t r;
            dlog << dlib::LINFO << "find-callback with search-string:" << search_expression;
            std::regex re(search_expression);
            auto match_end = std::sregex_iterator();
            for (auto const&tsn : ts_names) {
                if (std::sregex_iterator(tsn.begin(), tsn.end(), re)!= match_end) {
                    ts_info tsi; tsi.name = tsn;
                    r.push_back(tsi);
                }
            }
            return r;
        };

        server<standard_dtss_dispatcher> our_server(cb,fcb);

        // set up the server object we have made
        our_server.set_listening_ip("127.0.0.1");
        int port_no = 20000;
        our_server.set_listening_port(port_no);
        our_server.start_async();
        //while(our_server.is_running()&& our_server.get_listening_port()==0) //because dlib do not guarantee that listening port is set
        //    std::this_thread::sleep_for(std::chrono::milliseconds(3)); // upon return, so we have to wait until it's done

        //int port_no=our_server.get_listening_port();
        {
            string host_port = string("localhost:") + to_string(port_no);
            dlog << dlib::LINFO << "sending an expression ts to " << host_port ;
            std::vector<apoint_ts> tsl;
            for (size_t kb = 4;kb < 16;kb += 2)
                tsl.push_back(mk_expression(t, dt, kb * 1000)*apoint_ts(string("netcdf://group/path/ts") + std::to_string(kb)));
            client dtss(host_port);
            auto ts_b = dtss.evaluate(tsl, ta.total_period(),false,false);
            dlog << dlib::LINFO << "Got vector back, size= " << ts_b.size();
            for (const auto& ts : ts_b)
                dlog << dlib::LINFO << "ts.size()" << ts.size();
            dlog << dlib::LINFO << "testing 2 time:";
            FAST_REQUIRE_UNARY(our_server.is_running());
            dtss.evaluate(tsl, ta.period(0),false,false);
            dlog << dlib::LINFO << "done second test";
            // test search functions
            dlog << dlib::LINFO << "test .find function";
            auto found_ts = dtss.find(string("a.*"));
            FAST_REQUIRE_EQ(found_ts.size(), 1);
            FAST_CHECK_EQ(found_ts[0].name, ts_names[0]);
            dlog << dlib::LINFO << "test .find function done";

            throw_exception = true;// verify server-side exception gets back here.
            TS_ASSERT_THROWS_ANYTHING(dtss.evaluate(tsl, ta.period(0),false,false));
            dlog<<dlib::LINFO << "exceptions done,testing ordinary evaluate after exception";
            throw_exception = false;// verify server-side exception gets back here.
            dtss.evaluate(tsl, ta.period(0),false,false); // verify no exception here, should still work ok
            FAST_REQUIRE_UNARY(our_server.is_running());
            dlog << dlib::LINFO << "ok, -now testing percentiles";
            std::vector<int64_t> percentile_spec{ 0,25,50,-1,75,100 };
            auto percentiles = dtss.percentiles(tsl, ta.total_period(), ta24, percentile_spec,false,false);
            FAST_CHECK_EQ(percentiles.size(), percentile_spec.size());
            FAST_CHECK_EQ(percentiles[0].size(), ta24.size());
            dlog << dlib::LINFO << "done with percentiles, stopping localhost server";
            dtss.close();
            our_server.clear();
            dlog << dlib::LINFO << "done";
        }
    } catch (exception& e) {
        cout << e.what() << endl;
    }
    dlog << dlib::LINFO << "done";
}

TEST_CASE("dlib_multi_server_basics") {
    dlog.set_level(dlib::LALL);
    dlib::set_all_logging_output_streams(std::cout);
    dlib::set_all_logging_levels(dlib::LALL);
    dlog << dlib::LINFO << "Starting dtss multi-test";
    using namespace shyft::dtss;
    try {
        calendar utc;
        auto t = utc.time(2016, 1, 1);
        auto dt = deltahours(1);
        auto dt24 = deltahours(24);
        int n = 240;
        int n24 = n / 24;
        time_axis::fixed_dt ta(t, dt, n);
        gta_t ta24(t, dt24, n24);
        read_call_back_t rcb = [ta](id_vector_t ts_ids, core::utcperiod p)
            ->ts_vector_t {
            ts_vector_t r; r.reserve(ts_ids.size());
            double fv = 1.0;
            for (size_t i = 0; i < ts_ids.size(); ++i)
                r.emplace_back(ta, fv += 1.0);
            return r;
        };

        size_t n_servers=2;
        vector<unique_ptr<server<standard_dtss_dispatcher>>> servers;
        vector<string> host_ports;
        int base_port=21000;
        for(size_t i=0;i<n_servers;++i) {
            auto srv = make_unique<server<standard_dtss_dispatcher>>(rcb);
            srv->set_listening_ip("127.0.0.1");
            srv->set_listening_port(base_port +i);
            srv->start_async();
            servers.emplace_back(move(srv));
            host_ports.push_back(string("localhost:") + to_string(base_port + i));
        }
        {
            dlog << dlib::LINFO << "sending permutation of an expression sizes/vectors ts to "<<n_servers<< " hosts" ;
            client c(host_ports,false,1000);
            for(size_t n_ts=1;n_ts<10;n_ts+=1) {
                std::vector<apoint_ts> tsl;
                for (size_t kb = 1;kb <= n_ts;kb += 1) {
                    tsl.push_back(mk_expression(t, dt, (kb+1) * 10)*apoint_ts(string("netcdf://group/path/ts") + std::to_string(kb)));
                }
                //dlog << dlib::LINFO<< "Try with size " << tsl.size();
                auto ts_b = c.evaluate(tsl, ta.total_period(),false,false);
                //dlog << dlib::LINFO << "Got vector back, size= " << ts_b.size();
                //for (const auto& ts : ts_b)
                //    dlog << dlib::LINFO << "ts.size()" << ts.size();
                //dlog << dlib::LINFO << "-now testing percentiles";
                std::vector<int64_t> percentile_spec{ 0,25,50,-1,75,100 };
                auto percentiles = c.percentiles(tsl, ta.total_period(), ta24, percentile_spec,false,false);
                //dlog<<dlib::LINFO<<"done calc, verify";
                FAST_CHECK_EQ(percentiles.size(), percentile_spec.size());
                FAST_CHECK_EQ(percentiles[0].size(), ta24.size());
            }
            dlog << dlib::LINFO << "done with multi-server requests, stopping localhost server";
            c.close();
            for(size_t i =0;i<n_servers;++i) {
                dlog<<dlib::LINFO<<"Terminating server "<<i;
                servers[i]->clear();
            }
            dlog << dlib::LINFO << "done";
        }
    } catch (exception& e) {
        cout << e.what() << endl;
    }
    dlog << dlib::LINFO << "done";
}

TEST_CASE("dlib_server_performance") {
    dlog.set_level(dlib::LALL);
    dlib::set_all_logging_output_streams(std::cout);
    dlib::set_all_logging_levels(dlib::LALL);
    dlog << dlib::LINFO << "Starting dtss server performance test";
    using namespace shyft::dtss;
    try {
        // Tell the server to begin accepting connections.
        calendar utc;
        auto t = utc.time(2016, 1, 1);
        auto dt = deltahours(1);
        auto dt24 = deltahours(24);
        int n = 24 * 365 * 5;// 5years of hourly data
        int n24 = n / 24;
        int n_ts = 10;//83;
        gta_t ta(t, dt, n);
        gta_t ta24(t, dt24, n24);
        bool throw_exception = false;
        ts_vector_t from_disk; from_disk.reserve(n_ts);
        double fv = 1.0;
        for (int i = 0; i < n_ts; ++i)
            from_disk.emplace_back(ta, fv += 1.0,shyft::time_series::ts_point_fx::POINT_AVERAGE_VALUE);

        read_call_back_t cb = [&from_disk, &throw_exception](id_vector_t ts_ids, core::utcperiod p)
            ->ts_vector_t {
            if (throw_exception) {
                dlog << dlib::LINFO << "Throw from inside dtss executes!";
                throw std::runtime_error("test exception");
            }
            return from_disk;
        };
        server<standard_dtss_dispatcher> our_server(cb);

        // set up the server object we have made
        our_server.set_listening_ip("127.0.0.1");
        int port_no = 20000;
        our_server.set_listening_port(port_no);
        our_server.start_async();
        size_t n_threads = 1;

        vector<future<void>> clients;
        for (size_t i = 0;i < n_threads;++i) {
            clients.emplace_back(
                    async(launch::async, [port_no,ta,ta24,i,n_ts]()         /** thread this */ {
                    string host_port = string("localhost:") + to_string(port_no);
                    dlog << dlib::LINFO << "sending an expression ts to " << host_port;
                    std::vector<apoint_ts> tsl;
                    for (int x = 1; x <= n_ts; ++x) {// just make a  very thin request, that get loads of data back
#if 0
                        auto ts_expr = apoint_ts(string("netcdf://group/path/ts_") + std::to_string(x));
                        tsl.push_back(ts_expr);
#else
                        auto ts_expr = 10.0 + 3.0*apoint_ts(string("netcdf://group/path/ts_") + std::to_string(x));
                        if (x > 1) {
                            ts_expr = ts_expr - 3.0*apoint_ts(string("netcdf://group/path/ts_") + std::to_string(x - 1));
                        }
                        tsl.push_back(ts_expr.average(ta));
#endif
                    }

                    client dtss(host_port);
                    auto t0 = timing::now();
                    size_t eval_count = 0;
                    int test_duration_ms = 5000;
                    int kilo_points= tsl.size()*ta.size()/1000;
                    while (elapsed_ms(t0, timing::now()) < test_duration_ms) {
                        // burn cpu server side, save time on serialization
                        std::vector<int64_t> percentile_spec{ -1 };
                        auto percentiles = dtss.percentiles(tsl, ta.total_period(), ta24, percentile_spec,false,false);
                        //slower due to serialization:
                        //auto ts_b = dtss.evaluate(tsl, ta.total_period());
                        ++eval_count;
                    }
                    auto total_ms = double(elapsed_ms(t0, timing::now()));
                    dlog << dlib::LINFO << "Done testing " << i << ": #= " << eval_count
                    << " e/s = " << 1000 * double(eval_count) / total_ms << " [e/s]\n"
                    << " netw throughput = "<< kilo_points*eval_count*8/(total_ms)<< " Mb/sec";
                    dtss.close();
                }

                )
            );
        }
        dlog << dlib::LINFO << "waiting for test to complete";

        for (auto &f : clients) f.get();
        our_server.clear();
        dlog << dlib::LINFO << "done";
    } catch (exception& e) {
        cout << e.what() << endl;
        dlog<<dlib::LERROR<<"exception:"<<e.what();
    }
    dlog << dlib::LINFO << "done";
}
TEST_CASE("dtss_store_basics") {
        using namespace shyft::dtss;
        using namespace shyft::time_series::dd;
        using time_series::point_ts;

        std::shared_ptr<core::calendar> utc = std::make_shared<core::calendar>();
        std::shared_ptr<core::calendar> osl = std::make_shared<core::calendar>("Europe/Oslo");

        core::utctime t = utc->time(2016, 1, 1);
        core::utctimespan dt = core::deltahours(1);
        core::utctimespan dt_half = core::deltaminutes(30);
        std::size_t n = 24 * 365 * 2;//24*365*5;

        // construct time-axis that we want to test.
        time_axis::fixed_dt fta(t, dt, n);
        time_axis::calendar_dt cta1(utc,t,dt,n);
        time_axis::calendar_dt cta2(osl,t,dt,n);

        vector<utctime> tp;for(std::size_t i=0;i<fta.size();++i)tp.push_back(fta.time(i));
        time_axis::point_dt pta(tp,fta.total_period().end);
        dlog << dlib::LINFO << "starting store basics";

        auto dirname = "ts.db.test." + std::to_string(core::utctime_now());
        auto tmpdir = (fs::temp_directory_path()/dirname);
        ts_db db(tmpdir.string());

        TEST_SECTION("store_fixed_dt") {
            gts_t o(gta_t(fta),10.0,time_series::ts_point_fx::POINT_AVERAGE_VALUE);
            o.set(0, 100.); o.set(o.size()-1, 100.);
            std::string fn("measurements/tssf.db");  // verify we can have path-parts
            db.save(fn,o);

            // read all
            auto r1 = db.read(fn, utcperiod{});
            FAST_CHECK_EQ(o.point_interpretation(), r1.point_interpretation());
            FAST_CHECK_EQ(o.time_axis(), r1.time_axis());

            // read inner slice
            core::utctime tb = t + 3u*dt/2u;
            core::utctime te = t + (2u*n - 3u)*dt/2u;
            auto r2 = db.read(fn, utcperiod{ tb, te });
            FAST_CHECK_EQ(o.point_interpretation(), r2.point_interpretation());
            FAST_CHECK_EQ(r2.time_axis(), time_axis::generic_dt(t + dt, dt, n - 2u));
            FAST_CHECK_EQ(r2.value(0), o.value(1));  // dropped first value of o
            FAST_CHECK_EQ(r2.value(r2.size() - 1), o.value(o.size() - 2));  // dropped last value of o

            auto fr = db.find(string("measurements/.*\\.db")); // should match our ts.
            FAST_CHECK_EQ(fr.size(), 1 );

            db.remove(fn);
            fr = db.find(string("measurements/.*\\.db")); // should match our ts.
            FAST_CHECK_EQ(fr.size(),0);
        }

        TEST_SECTION("store_calendar_utc_dt") {
            gts_t o(gta_t(cta1), 10.0, time_series::ts_point_fx::POINT_AVERAGE_VALUE);
            o.set(0, 100.); o.set(o.size()-1, 100.);
            std::string fn("tssf1.db");
            db.save(fn, o);

            // read all
            auto r=db.read(fn, utcperiod{ });
            FAST_CHECK_EQ(o.point_interpretation(), r.point_interpretation());
            FAST_CHECK_EQ(o.time_axis(), r.time_axis());

            // read inner slice
            core::utctime tb = cta1.cal->add(t, dt_half, 3);
            core::utctime te = cta1.cal->add(t, dt_half, 2*n - 3);
            auto r2 = db.read(fn, utcperiod{ tb, te });

            FAST_CHECK_EQ(o.point_interpretation(), r2.point_interpretation());
            FAST_CHECK_EQ(r2.time_axis(), time_axis::generic_dt{ cta1.cal, cta1.cal->add(t, dt, 1), dt, n - 2u });
            FAST_CHECK_EQ(r2.value(0), o.value(1));  // dropped first value of o
            FAST_CHECK_EQ(r2.value(r2.size() - 1), o.value(o.size() - 2));  // dropped last value of o

            auto i = db.get_ts_info(fn);
            FAST_CHECK_EQ(i.name,fn);
            FAST_CHECK_EQ(i.data_period,o.total_period());
            FAST_CHECK_EQ(i.point_fx, o.point_interpretation());
            FAST_CHECK_LE( i.modified, utctime_now());

            auto fr = db.find(string(".ss.1\\.db")); // should match our ts.
            FAST_CHECK_EQ(fr.size(), 1 );

            db.remove(fn);
            try {
                auto rx=db.read(fn+".not.there", utcperiod{});
                FAST_CHECK_UNARY(rx.size()==3);
            } catch(const exception&) {
                FAST_CHECK_UNARY(true);
            }

        }
        TEST_SECTION("store_calendar_osl_dt") {
            gts_t o(gta_t(cta2),10.0,time_series::ts_point_fx::POINT_AVERAGE_VALUE);
            string fn("tssf2.db");
            db.save(fn,o);
            auto r=db.read(fn,utcperiod{});
            FAST_CHECK_EQ(o.point_interpretation(),r.point_interpretation());
            FAST_CHECK_EQ(o.time_axis(),r.time_axis());
            db.remove(fn);
        }
        TEST_SECTION("store_point_dt") {
            gts_t o(gta_t(pta), 10.0, time_series::ts_point_fx::POINT_INSTANT_VALUE);
            o.set(0, 100.); o.set(o.size()-1, 100.);
            string fn("tssf3.db");
            db.save(fn, o);

            // read all
            auto r = db.read(fn, utcperiod{});
            FAST_CHECK_EQ(o.point_interpretation(), r.point_interpretation());
            FAST_CHECK_EQ(o.time_axis(), r.time_axis());

            // read slice
            core::utctime tb = (pta.time(2) + pta.time(1)) / 2;
            core::utctime te = (pta.time(pta.size() - 1) + pta.time(pta.size() - 2)) / 2;
            auto r2 = db.read(fn, utcperiod{ tb, te });
            FAST_CHECK_EQ(o.point_interpretation(), r2.point_interpretation());
            time_axis::point_dt exp{ std::vector<core::utctime>( &o.ta.p.t[1], &o.ta.p.t[o.ta.p.t.size()-1] ), o.ta.p.t[o.ta.p.t.size()-1] };
            FAST_CHECK_EQ(r2.time_axis(), gta_t(exp));
            FAST_CHECK_EQ(r2.value(0), o.value(1));  // dropped first value of o
            FAST_CHECK_EQ(r2.value(r2.size() - 1), o.value(o.size() - 2));  // dropped last value of o

            auto i = db.get_ts_info(fn);
            FAST_CHECK_EQ(i.name,fn);
            FAST_CHECK_EQ(i.data_period,o.total_period());
            FAST_CHECK_EQ(i.point_fx, o.point_interpretation());
            FAST_CHECK_LE( i.modified, utctime_now());
            db.remove(fn);
        }

        TEST_SECTION("dtss_db_speed") {
            int n_ts = 120;
            vector<gts_t> tsv; tsv.reserve(n_ts);
            double fv = 1.0;
            for (int i = 0; i < n_ts; ++i)
                tsv.emplace_back(gta_t(fta), fv += 1.0,shyft::time_series::ts_point_fx::POINT_AVERAGE_VALUE);
            FAST_CHECK_EQ(n_ts,tsv.size());

            auto t0 = timing::now();
            for(std::size_t i=0;i<tsv.size();++i) {
                std::string fn("ts."+std::to_string(i)+".db");
                db.save(fn,tsv[i]);
            }
            auto t1 = timing::now();
            vector<gts_t> rv;
            for(std::size_t i=0;i<tsv.size();++i) {
                std::string fn("ts."+std::to_string(i)+".db");
                rv.push_back(db.read(fn,utcperiod{}));
            }
            auto t2= timing::now();
            auto w_mb_s= n_ts*n/double(elapsed_ms(t0,t1))/1000.0;
            auto r_mb_s= n_ts*n/double(elapsed_ms(t1,t2))/1000.0;
            // on windows(before workaround): ~ 6 mpts/sec write, 162 mpts/sec read (slow close->workaround with thread?)
            // on linux: ~ 120 mpts/sec write, 180 mpts/sec read
            std::cout<<"write Mpts/s = "<<w_mb_s<<", read Mpts/s = "<<r_mb_s<<" pts = "<<n_ts*n<<", roundtrip ms="<< double(elapsed_ms(t0,t2)) <<"\n";
            //std::cout << "open_ms:" << db.t_open << ", write_ms:" << db.t_write << ", t_close_ms:" << db.t_close << std::endl;
            FAST_CHECK_EQ(rv.size(),tsv.size());
            //fs::remove_all("*.db");
        }
#ifdef _WIN32
        for (int i = 0; i<10; ++i) {
            this_thread::sleep_for(chrono::duration<int, std::milli>(1000));
            try {
                fs::remove_all(tmpdir);
                break;
            }
            catch (...) {
                std::cout <<"Try #"<<i+1<< ":Failed to remove " << tmpdir << "\n";
            }
        }
#else
        fs::remove_all(tmpdir);
#endif

}
TEST_CASE("shyft_url") {

    using shyft::dtss::shyft_url;

    FAST_CHECK_EQ( shyft_url("abc","123"), string("shyft://abc/123") );

    std::map<std::string, std::string> m{ std::make_pair("foo", "bar"), std::make_pair("baz", "") };
    FAST_CHECK_EQ( shyft_url("abc","123", m), string("shyft://abc/123?baz=&foo=bar") );
}
TEST_CASE("extract_shyft_url_container") {

    using shyft::dtss::extract_shyft_url_container;

    std::string extracted_1 = extract_shyft_url_container("shyft://abc/something/else");
    std::string extracted_2 = extract_shyft_url_container("shyft://abc/something/else?query=string&here=foo");
    FAST_CHECK_EQ( extracted_1, string("abc") );
    FAST_CHECK_EQ( extracted_2, string("abc") );

    std::string extracted_3 = extract_shyft_url_container("grugge");
    std::string extracted_4 = extract_shyft_url_container("grugge?query");
    FAST_CHECK_EQ( extracted_3, string{} );
    FAST_CHECK_EQ( extracted_4, string{} );

}
TEST_CASE("extract_shyft_url_path") {

    using shyft::dtss::extract_shyft_url_path;

    std::string extracted_1 = extract_shyft_url_path("shyft://abc/something/else");
    std::string extracted_2 = extract_shyft_url_path("shyft://abc/something/else?query=string&here=foo");
    FAST_CHECK_EQ( extracted_1, string("something/else") );
    FAST_CHECK_EQ( extracted_2, string("something/else") );

    std::string extracted_3 = extract_shyft_url_path("grugge://abc/something/else");
    std::string extracted_4 = extract_shyft_url_path("grugge?query");
    FAST_CHECK_EQ( extracted_3, string{} );
    FAST_CHECK_EQ( extracted_4, string{} );

}
TEST_CASE("extract_shyft_url_query") {

    using shyft::dtss::extract_shyft_url_query_parameters;

    auto m1 = extract_shyft_url_query_parameters("shyft://abc/something/else?query=string&here=");
    FAST_CHECK_EQ( m1.size(), 2 );
    FAST_REQUIRE_EQ( m1.count("query"), 1 );
    FAST_CHECK_EQ( m1["query"], std::string{"string"} );
    FAST_REQUIRE_EQ( m1.count("here"), 1 );
    FAST_CHECK_EQ( m1["here"], std::string{""} );

    auto m2 = extract_shyft_url_query_parameters("shyft://abc/something/else?query=string&here=foo");
    FAST_CHECK_EQ( m2.size(), 2 );
    FAST_REQUIRE_EQ( m2.count("query"), 1 );
    FAST_CHECK_EQ( m2["query"], std::string{"string"} );
    FAST_REQUIRE_EQ( m2.count("here"), 1 );
    FAST_CHECK_EQ( m2["here"], std::string{"foo"} );

    auto m3 = extract_shyft_url_query_parameters("grugge");
    FAST_CHECK_EQ( m3.size(), 0 );

    auto m4 = extract_shyft_url_query_parameters("grugge?query");
    FAST_CHECK_EQ( m4.size(), 0 );
}
TEST_CASE("remove_shyft_url_queries") {

    using shyft::dtss::remove_shyft_url_queries;

    auto url1 = remove_shyft_url_queries("shyft://abc/something/else?query=string&here=");
    FAST_CHECK_EQ( url1, "shyft://abc/something/else" );

    auto url2 = remove_shyft_url_queries("grugge");
    FAST_CHECK_EQ( url2, "" );

    auto url3 = remove_shyft_url_queries("grugge?query");
    FAST_CHECK_EQ( url3, "" );
}
TEST_CASE("filter_shyft_url_parsed_queries") {

    using shyft::dtss::filter_shyft_url_parsed_queries;

    std::array<std::string, 2> to_remove{{ "key02", "key06" }};
    std::map<std::string, std::string> queries{{
        {"key01", "value01" }, {"key02", "value02" },
        {"key03", "value03" }, {"key04", "value04" },
        {"key05", "value05" }, {"key06", "value06" }
    }};

    filter_shyft_url_parsed_queries(queries, to_remove);

    FAST_CHECK_EQ( queries.size(), 4 );
    FAST_CHECK_EQ( queries.find("key02"), queries.cend() );
    FAST_CHECK_EQ( queries.find("key06"), queries.cend() );
}

TEST_CASE("dtss_store") { /*
    This test simply create and host a dtss on port 20000,
    then uses shyft:// prefix to test
    all internal operations that involve mapping to the
    shyft ts-db-store.
    */
    using namespace shyft::dtss;
    using namespace shyft::time_series::dd;
    using time_series::point_ts;
    using time_series::ts_point_fx;

    auto utc=make_shared<calendar>();
    auto t = utc->time(2016, 1, 1);
    auto dt = deltahours(1);
    int n = 24 * 365 * 2;//24*365*5;

    // make dtss server
    auto tmpdir = fs::temp_directory_path()/"shyft.c.test";
    server<standard_dtss_dispatcher> our_server{};
    string tc{"tc"};
    our_server.add_container(tc,tmpdir.string());
    our_server.set_listening_ip("127.0.0.1");
    int port_no = 20000;
    string host_port = string("localhost:") + to_string(port_no);

    our_server.set_listening_port(port_no);
    our_server.start_async();
    // make corresponding client that we will use for the test.
    client dtss(host_port);
    SUBCASE("save_find_read") {
        size_t n_ts=10;
        time_axis::fixed_dt fta(t, dt, n);
        time_axis::generic_dt gta{t,dt*24,size_t(n/24)};
        const auto stair_case=ts_point_fx::POINT_AVERAGE_VALUE;
        ts_vector_t tsv;
        vector<point_ts<time_axis::fixed_dt>> ftsv;

        for(size_t i=0;i<n_ts;++i) {
            tsv.emplace_back(
                    shyft_url(tc,to_string(i)),
                    apoint_ts{fta,i*10.0,stair_case}
            );
            ftsv.emplace_back(fta,i*10.0,stair_case);
        }
        auto f0 = dtss.find(shyft_url(tc,".*"));
        FAST_CHECK_EQ(f0.size(),0);// expect zero to start with
        auto t0=timing::now();
        dtss.store_ts(tsv, true, false);
        auto t1=timing::now();
        auto f1 = dtss.find(shyft_url(tc,".*"));
        FAST_CHECK_EQ(f1.size(),tsv.size());
        ts_vector_t ev;
        for(size_t i=0;i<tsv.size();++i)
            ev.push_back(
                         3.0*apoint_ts(shyft_url(tc,to_string(i)))
                           //+ apoint_ts(shyft_url(tc,to_string(i>0?i-1:i)))
                         );
        // activate auto-cache, to prepare for next
        our_server.set_auto_cache(true);
        vector<int> pc{10,50,90};
        auto t2 = timing::now();
        //auto er= dtss.percentiles(ev,fta.total_period(),gta,pc);//uncached read
        auto er= dtss.evaluate(ev,fta.total_period(),true,true);//uncached read
        auto t3 = timing::now();
        //auto ec = dtss.percentiles(ev, fta.total_period(),gta,pc);
        auto ec = dtss.evaluate(ev, fta.total_period(),true,true);
        auto t4 = timing::now();// cached read.
        //-- establish benchmark
        vector<vector<double>> bmr;bmr.reserve(n_ts);
        for(const auto &ts:ftsv) {
            auto calc= 3.0* ts;
            vector<double> r;r.reserve(calc.size());
            for(size_t i=0;i<calc.size();++i)
                r.emplace_back(calc.value(i));
            bmr.emplace_back(move(r));
        }
        auto t5 =timing::now();
        FAST_CHECK_EQ(bmr.size(),n_ts);
//        FAST_CHECK_EQ(er.size(),pc.size());
//        FAST_CHECK_EQ(ec.size(), pc.size());
        FAST_CHECK_EQ(er.size(),ev.size());
        FAST_CHECK_EQ(ec.size(), ev.size());
        std::cout<<"store mpts/s "<<double(n_ts*n)/(double(elapsed_us(t0,t1))/1000000.0)/1e6<<"\n";
        std::cout<<"evalr mpts/s "<<double(n_ts*n)/(double(elapsed_us(t2,t3))/1000000.0)/1e6<<"\n";
        std::cout<<"evalc mpts/s "<<double(n_ts*n)/(double(elapsed_us(t3,t4))/1000000.0)/1e6<<"\n";
        std::cout<<"bench mpts/s "<<double(n_ts*n)/(double(elapsed_us(t4,t5))/1000000.0)/1e6<<"\t time :"<<double(elapsed_ms(t4,t5))<<"\n";
        auto cs = our_server.get_cache_stats();
        std::cout<<"cache stats(hits,misses,cover_misses,id_count,frag_count,point_count):\n "<<cs.hits<<","<<cs.misses<<","<<cs.coverage_misses<<","<<cs.id_count<<","<<cs.fragment_count<<","<<cs.point_count<<")\n";
    }

    our_server.clear();
#ifdef _WIN32
    for (int i = 0; i<10; ++i) {
        this_thread::sleep_for(chrono::duration<int, std::milli>(1000));
        try {
            fs::remove_all(tmpdir);
            break;
        }
        catch (...) {
            std::cout<<"Try #"<<i+1 << ": Failed to remove " << tmpdir << "\n";
        }
    }
#else
    fs::remove_all(tmpdir);
#endif

}

TEST_CASE("dtss_store_merge_write") {

    namespace core = shyft::core;
    namespace dtss = shyft::dtss;
    namespace ta = shyft::time_axis;
    namespace ts = shyft::time_series;
    using shyft::time_series::dd::gta_t;
    // setup db
    auto dirname = "ts.db.test." + std::to_string(core::utctime_now());
    auto tmpdir = (fs::temp_directory_path()/dirname);
    dtss::ts_db db(tmpdir.string());

    SUBCASE("error_handling") {
        SUBCASE("extending with different ta") {
            // data
            std::shared_ptr<core::calendar> utc_ptr = std::make_shared<core::calendar>();
            const core::utctime t0 = core::utctime_now();
            const core::utctimespan dt_h = core::calendar::HOUR;
            const core::utctimespan dt_d = core::calendar::DAY;
            const std::size_t n = 1000;
            // -----
            ta::fixed_dt f_ta_h{ t0, dt_h, n };
            ta::calendar_dt c_ta_d{ utc_ptr, t0, dt_d, n };
            ts::point_ts<ta::generic_dt> pts_h{ gta_t(f_ta_h), 0. };
            ts::point_ts<ta::generic_dt> pts_d{ gta_t(c_ta_d), 0. };
            // -----
            std::string fn("dtss_save_merge/ext_diff_ta.db");

            // save initital data
            db.save(fn, pts_d, false);
            auto find_res = db.find(string("dtss_save_merge/ext_diff_ta\\.db"));
            FAST_CHECK_EQ(find_res.size(), 1);

            // add data to the same path
            CHECK_THROWS_AS_MESSAGE(db.save(fn, pts_h, false),
                std::runtime_error, "dtss_store: cannot merge with different ta type");

            // cleanup
            db.remove(fn);
            find_res = db.find(string("dtss_save_merge/ext_diff_ta\\.db")); // should match our ts.
            FAST_CHECK_EQ(find_res.size(), 0);
        }
        SUBCASE("extending with different point interpretation") {
            // data
            std::shared_ptr<core::calendar> utc_ptr = std::make_shared<core::calendar>();
            const core::utctime t0 = core::utctime_now();
            const core::utctimespan dt_h = core::calendar::HOUR;
            const core::utctimespan dt_d = core::calendar::DAY;
            const std::size_t n = 1000;
            // -----
            ta::fixed_dt f_ta_h{ t0, dt_h, n };
            ta::fixed_dt f_ta_d{ t0, dt_d, n };
            ts::point_ts<ta::generic_dt> pts_h{ gta_t(f_ta_h), 0., ts::POINT_INSTANT_VALUE };
            ts::point_ts<ta::generic_dt> pts_d{ gta_t(f_ta_d), 0., ts::POINT_AVERAGE_VALUE };
            // -----
            std::string fn("dtss_save_merge/ext_diff_fx.db");

            // save initital data
            db.save(fn, pts_d, false);
            auto find_res = db.find(string("dtss_save_merge/ext_diff_fx\\.db"));
            FAST_REQUIRE_EQ(find_res.size(), 1);

            // add data to the same path
            CHECK_THROWS_AS_MESSAGE(db.save(fn, pts_h, false),
                std::runtime_error, "dtss_store: cannot merge with different point interpretation");

            // cleanup
            db.remove(fn);
            find_res = db.find(string("dtss_save_merge/ext_diff_fx\\.db")); // should match our ts.
            FAST_CHECK_EQ(find_res.size(), 0);
        }
        SUBCASE("fixed_dt old_dt != new_dt") {
            // data
            const core::utctime t0 = core::utctime_now();
            const core::utctimespan dt_h = core::calendar::HOUR;
            const core::utctimespan dt_d = core::calendar::DAY;
            const std::size_t n = 1000;
            // -----
            ta::fixed_dt f_ta_h{ t0, dt_h, n };
            ta::fixed_dt f_ta_d{ t0, dt_d, n };
            ts::point_ts<ta::generic_dt> pts_h{ gta_t(f_ta_h), 0. };
            ts::point_ts<ta::generic_dt> pts_d{ gta_t(f_ta_d), 0. };
            // -----
            std::string fn("dtss_save_merge/ext_fixed_diff_dt.db");  // verify we can have path-parts

            // save initital data
            db.save(fn, pts_d, false);
            auto find_res = db.find(string("dtss_save_merge/ext_fixed_diff_dt\\.db"));
            FAST_CHECK_EQ(find_res.size(), 1);

            // add data to the same path
            CHECK_THROWS_AS_MESSAGE(db.save(fn, pts_h, false),
                std::runtime_error, "dtss_store: cannot merge unaligned fixed_dt");

            // cleanup
            db.remove(fn);
            find_res = db.find(string("dtss_save_merge/ext_fixed_diff_dt\\.db")); // should match our ts.
            FAST_CHECK_EQ(find_res.size(), 0);
        }
        SUBCASE("fixed_dt unaligned axes") {
            // data
            core::calendar utc{};
            const core::utctime t0_1 = utc.time(2000, 1, 1, 0, 0);
            const core::utctime t0_2 = utc.time(2000, 1, 1, 0, 13);  // 13 minutes shifted
            const core::utctimespan dt_h = core::calendar::HOUR;
            const std::size_t n = 1000;
            // -----
            ta::fixed_dt f_ta_h{ t0_1, dt_h, n };
            ta::fixed_dt f_ta_d{ t0_2, dt_h, n };
            ts::point_ts<ta::generic_dt> pts_h{ gta_t(f_ta_h), 0. };
            ts::point_ts<ta::generic_dt> pts_d{ gta_t(f_ta_d), 0. };
            // -----
            std::string fn("dtss_save_merge/ext_fixed_unaligned.db");  // verify we can have path-parts

                                                     // save initital data
            db.save(fn, pts_d, false);
            auto find_res = db.find(string("dtss_save_merge/ext_fixed_unaligned\\.db"));
            FAST_CHECK_EQ(find_res.size(), 1);

            // add data to the same path
            CHECK_THROWS_AS_MESSAGE(db.save(fn, pts_h, false),
                std::runtime_error, "dtss_store: cannot merge unaligned fixed_dt");

            // cleanup
            db.remove(fn);
            find_res = db.find(string("dtss_save_merge/ext_fixed_unaligned\\.db")); // should match our ts.
            FAST_CHECK_EQ(find_res.size(), 0);
        }
        SUBCASE("calendar_dt old_dt != new_dt") {
            // data
            std::shared_ptr<core::calendar> utc_ptr = std::make_shared<core::calendar>();
            const core::utctime t0 = core::utctime_now();
            const core::utctimespan dt_h = core::calendar::HOUR;
            const core::utctimespan dt_d = core::calendar::DAY;
            const std::size_t n = 1000;
            // -----
            ta::calendar_dt f_ta_h{ utc_ptr, t0, dt_h, n };
            ta::calendar_dt f_ta_d{ utc_ptr, t0, dt_d, n };
            ts::point_ts<ta::generic_dt> pts_h{ gta_t(f_ta_h), 0. };
            ts::point_ts<ta::generic_dt> pts_d{ gta_t(f_ta_d), 0. };
            // -----
            std::string fn("dtss_save_merge/ext_cal_diff_dt.db");  // verify we can have path-parts

           // save initital data
            db.save(fn, pts_d, false);
            auto find_res = db.find(string("dtss_save_merge/ext_cal_diff_dt\\.db"));
            FAST_CHECK_EQ(find_res.size(), 1);

            // add data to the same path
            CHECK_THROWS_AS_MESSAGE(db.save(fn, pts_h, false),
                std::runtime_error, "dtss_store: cannot merge unaligned calendar_dt");

            // cleanup
            db.remove(fn);
            find_res = db.find(string("dtss_save_merge/ext_cal_diff_dt\\.db")); // should match our ts.
            FAST_CHECK_EQ(find_res.size(), 0);
        }
        SUBCASE("calendar_dt unaligned axes") {
            // data
            std::shared_ptr<core::calendar> utc_ptr = std::make_shared<core::calendar>();
            const core::utctime t0_1 = utc_ptr->time(2000, 1, 1, 0, 0);
            const core::utctime t0_2 = utc_ptr->time(2000, 1, 1, 0, 13);  // 13 minutes shifted
            const core::utctimespan dt_h = core::calendar::HOUR;
            const std::size_t n = 1000;
            // -----
            ta::calendar_dt f_ta_h{ utc_ptr, t0_1, dt_h, n };
            ta::calendar_dt f_ta_d{ utc_ptr, t0_2, dt_h, n };
            ts::point_ts<ta::generic_dt> pts_h{ gta_t(f_ta_h), 0. };
            ts::point_ts<ta::generic_dt> pts_d{ gta_t(f_ta_d), 0. };
            // -----
            std::string fn("dtss_save_merge/ext_cal_unaligned.db");  // verify we can have path-parts

            // save initital data
            db.save(fn, pts_d, false);
            auto find_res = db.find(string("dtss_save_merge/ext_cal_unaligned\\.db"));
            FAST_CHECK_EQ(find_res.size(), 1);

            // add data to the same path
            CHECK_THROWS_AS_MESSAGE(db.save(fn, pts_h, false),
                std::runtime_error, "dtss_store: cannot merge unaligned calendar_dt");

            // cleanup
            db.remove(fn);
            find_res = db.find(string("dtss_save_merge/ext_cal_unaligned\\.db")); // should match our ts.
            FAST_CHECK_EQ(find_res.size(), 0);
        }
        SUBCASE("calendar_dt different calendars") {
            // data
            std::shared_ptr<core::calendar> utc_ptr = std::make_shared<core::calendar>();
            std::shared_ptr<core::calendar> osl_ptr = std::make_shared<core::calendar>("Europe/Oslo");
            const core::utctime t0_1 = utc_ptr->time(2000, 1, 1, 0, 0);
            const core::utctime t0_2 = osl_ptr->time(2000, 1, 1, 0, 13);  // 13 minutes shifted
            const core::utctimespan dt_h = core::calendar::HOUR;
            const std::size_t n = 1000;
            // -----
            ta::calendar_dt f_ta_h{ utc_ptr, t0_1, dt_h, n };
            ta::calendar_dt f_ta_d{ osl_ptr, t0_2, dt_h, n };
            ts::point_ts<ta::generic_dt> pts_h{ gta_t(f_ta_h), 0. };
            ts::point_ts<ta::generic_dt> pts_d{ gta_t(f_ta_d), 0. };
            // -----
            std::string fn("dtss_save_merge/ext_cal_diff_cal.db");  // verify we can have path-parts

            // save initital data
            db.save(fn, pts_d, false);
            auto find_res = db.find(string("dtss_save_merge/ext_cal_diff_cal\\.db"));
            FAST_CHECK_EQ(find_res.size(), 1);

            // add data to the same path
            CHECK_THROWS_AS_MESSAGE(db.save(fn, pts_h, false),
                std::runtime_error, "dtss_store: cannot merge calendar_dt with different calendars");

            // cleanup
            db.remove(fn);
            find_res = db.find(string("dtss_save_merge/ext_cal_diff_cal\\.db")); // should match our ts.
            FAST_CHECK_EQ(find_res.size(), 0);
        }
    }
    SUBCASE("merging time-series") {
        std::shared_ptr<core::calendar> utc_ptr = std::make_shared<core::calendar>();
        std::shared_ptr<core::calendar> osl_ptr = std::make_shared<core::calendar>("Europe/Oslo");

        SUBCASE("fixed_dt") {
            SUBCASE("exact") {
                // data
                const core::utctimespan dt = core::calendar::DAY;
                const std::size_t n = 100;
                const core::utctime t0 = utc_ptr->time(2016, 1, 1);
                // -----
                ta::fixed_dt f_ta{ t0, dt, n };
                ts::point_ts<ta::generic_dt> pts_old{ gta_t(f_ta), 1. };
                ts::point_ts<ta::generic_dt> pts_new{ gta_t(f_ta), 10. };
                // -----
                std::string fn("dtss_save_merge/fixed_exact.db");

                // save initital data
                db.save(fn, pts_old, false);
                auto find_res = db.find("dtss_save_merge/fixed_exact\\.db");
                FAST_CHECK_EQ(find_res.size(), 1);

                // add data to the same path
                db.save(fn, pts_new, false);

                // check merged data
                ts::point_ts<ta::generic_dt> res = db.read("dtss_save_merge/fixed_exact.db", core::utcperiod{ });
                // time-axis
                FAST_CHECK_EQ(res.ta.gt, time_axis::generic_dt::FIXED);
                FAST_CHECK_EQ(res.total_period().start, f_ta.total_period().start);
                FAST_CHECK_EQ(res.total_period().end, f_ta.total_period().end);
                // values
                FAST_CHECK_EQ(res.v.at(0), 10.);
                FAST_CHECK_EQ(res.v.at(n - 1), 10.);

                // cleanup
                db.remove(fn);
                find_res = db.find(string("dtss_save_merge/fixed_exact\\.db"));
                FAST_CHECK_EQ(find_res.size(), 0);
            }
            SUBCASE("new contained in old") {
                // data
                const std::size_t drop = 10u;  // points to drop from start/end of old
                // -----
                const core::utctimespan dt = core::calendar::DAY;
                const std::size_t n = 100;
                const core::utctime t0_old = utc_ptr->time(2016, 1, 1);
                const core::utctime t0_new = t0_old + dt*drop;
                // -----
                ta::fixed_dt f_ta_old{ t0_old, dt, n };
                ta::fixed_dt f_ta_new{ t0_new, dt, n - 2*drop };
                ts::point_ts<ta::generic_dt> pts_old{ gta_t(f_ta_old), 1. };
                ts::point_ts<ta::generic_dt> pts_new{ gta_t(f_ta_new), 10. };
                // -----
                std::string fn("dtss_save_merge/fixed_new_in_old.db");

                // save initital data
                db.save(fn, pts_old, false);
                auto find_res = db.find("dtss_save_merge/fixed_new_in_old\\.db");
                FAST_CHECK_EQ(find_res.size(), 1);

                // add data to the same path
                db.save(fn, pts_new, false);

                // check merged data
                ts::point_ts<ta::generic_dt> res = db.read("dtss_save_merge/fixed_new_in_old.db", core::utcperiod{ });
                // time-axis
                FAST_CHECK_EQ(res.ta.gt, time_axis::generic_dt::FIXED);
                FAST_CHECK_EQ(res.total_period().start, f_ta_old.total_period().start);
                FAST_CHECK_EQ(res.total_period().end, f_ta_old.total_period().end);
                // values
                FAST_CHECK_EQ(res.v.at(0), 1.);
                FAST_CHECK_EQ(res.v.at(drop - 1), 1.);
                FAST_CHECK_EQ(res.v.at(drop), 10.);
                FAST_CHECK_EQ(res.v.at(n - drop - 1), 10.);
                FAST_CHECK_EQ(res.v.at(n - drop), 1.);
                FAST_CHECK_EQ(res.v.at(n - 1), 1.);

                // cleanup
                db.remove(fn);
                find_res = db.find(string("dtss_save_merge/fixed_new_in_old\\.db"));
                FAST_CHECK_EQ(find_res.size(), 0);
            }
            SUBCASE("old contained in new") {
                // data
                const std::size_t extra = 10u;  // points to drop from start/end of old
                // -----
                const core::utctimespan dt = core::calendar::DAY;
                const std::size_t n = 100;
                const core::utctime t0_old = utc_ptr->time(2016, 1, 1);
                const core::utctime t0_new = t0_old - dt*extra;
                // -----
                ta::fixed_dt f_ta_old{ t0_old, dt, n };
                ta::fixed_dt f_ta_new{ t0_new, dt, n + extra };
                ts::point_ts<ta::generic_dt> pts_old{ gta_t(f_ta_old), 1. };
                ts::point_ts<ta::generic_dt> pts_new{ gta_t(f_ta_new), 10. };
                // -----
                std::string fn("dtss_save_merge/fixed_old_in_new.db");

                // save initital data
                db.save(fn, pts_old, false);
                auto find_res = db.find("dtss_save_merge/fixed_old_in_new\\.db");
                FAST_CHECK_EQ(find_res.size(), 1);

                // add data to the same path
                db.save(fn, pts_new, false);

                // check merged data
                ts::point_ts<ta::generic_dt> res = db.read("dtss_save_merge/fixed_old_in_new.db", core::utcperiod{ });
                // time-axis
                FAST_CHECK_EQ(res.ta.gt, time_axis::generic_dt::FIXED);
                FAST_CHECK_EQ(res.total_period().start, f_ta_new.total_period().start);
                FAST_CHECK_EQ(res.total_period().end, f_ta_new.total_period().end);
                // values
                FAST_CHECK_EQ(res.v.at(0), 10.);
                FAST_CHECK_EQ(res.v.at(n - 1), 10.);

                // cleanup
                db.remove(fn);
                find_res = db.find(string("dtss_save_merge/fixed_old_in_new\\.db"));
                FAST_CHECK_EQ(find_res.size(), 0);
            }
            SUBCASE("new overlap start of old") {
                // data
                const core::utctimespan dt = core::calendar::DAY;
                const std::size_t n = 100;
                const core::utctime t0_old = utc_ptr->time(2016, 1, 1);
                const core::utctime t0_new = t0_old - dt*n/2;
                // -----
                ta::fixed_dt f_ta_old{ t0_old, dt, n };
                ta::fixed_dt f_ta_new{ t0_new, dt, n };
                ts::point_ts<ta::generic_dt> pts_old{ gta_t(f_ta_old), 1. };
                ts::point_ts<ta::generic_dt> pts_new{ gta_t(f_ta_new), 10. };
                // -----
                std::string fn("dtss_save_merge/fixed_new_over_start.db");

                FAST_REQUIRE_LT(t0_new, t0_old);

                // save initital data
                db.save(fn, pts_old, false);
                auto find_res = db.find("dtss_save_merge/fixed_new_over_start\\.db");
                FAST_CHECK_EQ(find_res.size(), 1);

                // add data to the same path
                db.save(fn, pts_new, false);

                // check merged data
                ts::point_ts<ta::generic_dt> res = db.read("dtss_save_merge/fixed_new_over_start.db", core::utcperiod{ });
                // time-axis
                FAST_CHECK_EQ(res.ta.gt, time_axis::generic_dt::FIXED);
                FAST_CHECK_EQ(res.total_period().start, f_ta_new.total_period().start);
                FAST_CHECK_EQ(res.total_period().end, f_ta_old.total_period().end);
                // values
                FAST_CHECK_EQ(res.v.at(0), 10.);
                FAST_CHECK_EQ(res.v.at(n - 1), 10.);
                FAST_CHECK_EQ(res.v.at(n), 1.);
                FAST_CHECK_EQ(res.v.at(n + n/2 - 1), 1.);

                // cleanup
                db.remove(fn);
                find_res = db.find(string("dtss_save_merge/fixed_new_over_start\\.db"));
                FAST_CHECK_EQ(find_res.size(), 0);
            }
            SUBCASE("new overlap end of old") {
                // data
                const core::utctimespan dt = core::calendar::DAY;
                const std::size_t n = 100;
                const core::utctime t0_old = utc_ptr->time(2016, 1, 1);
                const core::utctime t0_new = t0_old + dt*n/2;
                // -----
                ta::fixed_dt f_ta_old{ t0_old, dt, n };
                ta::fixed_dt f_ta_new{ t0_new, dt, n };
                ts::point_ts<ta::generic_dt> pts_old{ gta_t(f_ta_old), 1. };
                ts::point_ts<ta::generic_dt> pts_new{ gta_t(f_ta_new), 10. };
                // -----
                std::string fn("dtss_save_merge/fixed_new_over_end.db");

                // save initital data
                db.save(fn, pts_old, false);
                auto find_res = db.find("dtss_save_merge/fixed_new_over_end\\.db");
                FAST_CHECK_EQ(find_res.size(), 1);

                // add data to the same path
                db.save(fn, pts_new, false);

                // check merged data
                ts::point_ts<ta::generic_dt> res = db.read("dtss_save_merge/fixed_new_over_end.db", core::utcperiod{ });
                // time-axis
                FAST_CHECK_EQ(res.ta.gt, time_axis::generic_dt::FIXED);
                FAST_CHECK_EQ(res.total_period().start, f_ta_old.total_period().start);
                FAST_CHECK_EQ(res.total_period().end, f_ta_new.total_period().end);
                // values
                FAST_CHECK_EQ(res.v.at(0), 1.);
                FAST_CHECK_EQ(res.v.at(n/2 - 1), 1.);
                FAST_CHECK_EQ(res.v.at(n/2), 10.);
                FAST_CHECK_EQ(res.v.at(n + n/2 - 1), 10.);

                // cleanup
                db.remove(fn);
                find_res = db.find(string("dtss_save_merge/fixed_new_over_end\\.db"));
                FAST_CHECK_EQ(find_res.size(), 0);
            }
            SUBCASE("consecutive without gap") {
                // data
                const core::utctimespan dt = core::calendar::DAY;
                const std::size_t n = 100;
                const core::utctime t0_old = utc_ptr->time(2016, 1, 1);
                const core::utctime t0_new = t0_old + dt*n;
                // -----
                ta::fixed_dt f_ta_old{ t0_old, dt, n };
                ta::fixed_dt f_ta_new{ t0_new, dt, n };
                ts::point_ts<ta::generic_dt> pts_old{ gta_t(f_ta_old), 1. };
                ts::point_ts<ta::generic_dt> pts_new{ gta_t(f_ta_new), 10. };
                // -----
                std::string fn("dtss_save_merge/fixed_consec.db");

                // save initital data
                db.save(fn, pts_old, false);
                auto find_res = db.find("dtss_save_merge/fixed_consec\\.db");
                FAST_CHECK_EQ(find_res.size(), 1);

                // add data to the same path
                db.save(fn, pts_new, false);

                // check merged data
                ts::point_ts<ta::generic_dt> res = db.read("dtss_save_merge/fixed_consec.db", core::utcperiod{ });
                // time-axis
                FAST_CHECK_EQ(res.ta.gt, time_axis::generic_dt::FIXED);
                FAST_CHECK_EQ(res.total_period().start, f_ta_old.total_period().start);
                FAST_CHECK_EQ(res.total_period().end, f_ta_new.total_period().end);
                // values
                FAST_CHECK_EQ(res.v.at(0), 1.);
                FAST_CHECK_EQ(res.v.at(n - 1), 1.);
                FAST_CHECK_EQ(res.v.at(n), 10.);
                FAST_CHECK_EQ(res.v.at(2 * n - 1), 10.);

                // cleanup
                db.remove(fn);
                find_res = db.find(string("dtss_save_merge/fixed_consec\\.db"));
                FAST_CHECK_EQ(find_res.size(), 0);
            }
            SUBCASE("new after old with gap") {
                // data
                const core::utctimespan dt = core::calendar::DAY;
                const std::size_t n = 100;
                const core::utctime t0_old = utc_ptr->time(2016, 1, 1);
                const core::utctime t0_new = t0_old + 2*n*dt;
                // -----
                ta::fixed_dt f_ta_old{ t0_old, dt, n };
                ta::fixed_dt f_ta_new{ t0_new, dt, n };
                ts::point_ts<ta::generic_dt> pts_old{ gta_t(f_ta_old), 1. };
                ts::point_ts<ta::generic_dt> pts_new{ gta_t(f_ta_new), 10. };
                // -----
                std::string fn("dtss_save_merge/fixed_gap_after.db");

                // save initital data
                db.save(fn, pts_old, false);
                auto find_res = db.find("dtss_save_merge/fixed_gap_after\\.db");
                FAST_CHECK_EQ(find_res.size(), 1);

                // add data to the same path
                db.save(fn, pts_new, false);

                // check merged data
                ts::point_ts<ta::generic_dt> res = db.read("dtss_save_merge/fixed_gap_after.db", core::utcperiod{ });
                // time-axis
                FAST_CHECK_EQ(res.ta.gt, time_axis::generic_dt::FIXED);
                FAST_CHECK_EQ(res.ta.size(), 3*n);
                FAST_CHECK_EQ(res.total_period().start, f_ta_old.total_period().start);
                FAST_CHECK_EQ(res.total_period().end, f_ta_new.total_period().end);
                // values
                FAST_CHECK_EQ(res.v.at(0), 1.);
                FAST_CHECK_EQ(res.v.at(n - 1), 1.);
                FAST_CHECK_UNARY(std::isnan(res.v.at(n)));
                FAST_CHECK_UNARY(std::isnan(res.v.at(2*n - 1)));
                FAST_CHECK_EQ(res.v.at(2 * n), 10.);
                FAST_CHECK_EQ(res.v.at(3 * n - 1), 10.);

                // cleanup
                db.remove(fn);
                find_res = db.find(string("dtss_save_merge/fixed_gap_after\\.db"));
                FAST_CHECK_EQ(find_res.size(), 0);
            }
            SUBCASE("new before old with gap") {
                // data
                const core::utctimespan dt = core::calendar::DAY;
                const std::size_t n = 100;
                const core::utctime t0_old = utc_ptr->time(2016, 1, 1);
                const core::utctime t0_new = t0_old - 2*n*dt;
                // -----
                ta::fixed_dt f_ta_old{ t0_old, dt, n };
                ta::fixed_dt f_ta_new{ t0_new, dt, n };
                ts::point_ts<ta::generic_dt> pts_old{ gta_t(f_ta_old), 1. };
                ts::point_ts<ta::generic_dt> pts_new{ gta_t(f_ta_new), 10. };
                // -----
                std::string fn("dtss_save_merge/fixed_gap_before.db");

                // save initital data
                db.save(fn, pts_old, false);
                auto find_res = db.find("dtss_save_merge/fixed_gap_before\\.db");
                FAST_CHECK_EQ(find_res.size(), 1);

                // add data to the same path
                db.save(fn, pts_new, false);

                // check merged data
                ts::point_ts<ta::generic_dt> res = db.read("dtss_save_merge/fixed_gap_before.db", core::utcperiod{ });
                // time-axis
                FAST_CHECK_EQ(res.ta.gt, time_axis::generic_dt::FIXED);
                FAST_CHECK_EQ(res.ta.size(), 3*n);
                FAST_CHECK_EQ(res.total_period().start, f_ta_new.total_period().start);
                FAST_CHECK_EQ(res.total_period().end, f_ta_old.total_period().end);
                // values
                FAST_CHECK_EQ(res.v.at(0), 10.);
                FAST_CHECK_EQ(res.v.at(n - 1), 10.);
                FAST_CHECK_UNARY(std::isnan(res.v.at(n)));
                FAST_CHECK_UNARY(std::isnan(res.v.at(2*n - 1)));
                FAST_CHECK_EQ(res.v.at(2 * n), 1.);
                FAST_CHECK_EQ(res.v.at(3 * n - 1), 1.);

                // cleanup
                db.remove(fn);
                find_res = db.find(string("dtss_save_merge/fixed_gap_before\\.db"));
                FAST_CHECK_EQ(find_res.size(), 0);
            }
        }
        SUBCASE("calendar_dt") {
            SUBCASE("exact") {
                // data
                const core::utctimespan dt = core::calendar::DAY;
                const std::size_t n = 100;
                const core::utctime t0 = utc_ptr->time(2016, 1, 1);
                // -----
                ta::calendar_dt c_ta{ utc_ptr, t0, dt, n };
                ts::point_ts<ta::generic_dt> pts_old{ gta_t(c_ta), 1. };
                ts::point_ts<ta::generic_dt> pts_new{ gta_t(c_ta), 10. };
                // -----
                std::string fn("dtss_save_merge/calendar_exact.db");

                // save initital data
                db.save(fn, pts_old, false);
                auto find_res = db.find("dtss_save_merge/calendar_exact\\.db");
                FAST_CHECK_EQ(find_res.size(), 1);

                // add data to the same path
                db.save(fn, pts_new, false);

                // check merged data
                ts::point_ts<ta::generic_dt> res = db.read("dtss_save_merge/calendar_exact.db", core::utcperiod{ });
                // time-axis
                FAST_CHECK_EQ(res.ta.gt, time_axis::generic_dt::CALENDAR);
                FAST_CHECK_EQ(res.total_period().start, c_ta.total_period().start);
                FAST_CHECK_EQ(res.total_period().end, c_ta.total_period().end);
                // values
                FAST_CHECK_EQ(res.v.at(0), 10.);
                FAST_CHECK_EQ(res.v.at(n - 1), 10.);

                // cleanup
                db.remove(fn);
                find_res = db.find(string("dtss_save_merge/calendar_exact\\.db"));
                FAST_CHECK_EQ(find_res.size(), 0);
            }
            SUBCASE("new contained in old") {
                // data
                const std::size_t drop = 10u;  // points to drop from start/end of old
                // -----
                const core::utctimespan dt = core::calendar::DAY;
                const std::size_t n = 100;
                const core::utctime t0_old = utc_ptr->time(2016, 1, 1);
                const core::utctime t0_new = t0_old + dt*drop;
                // -----
                ta::calendar_dt c_ta_old{ utc_ptr, t0_old, dt, n };
                ta::calendar_dt c_ta_new{ utc_ptr, t0_new, dt, n - 2*drop };
                ts::point_ts<ta::generic_dt> pts_old{ gta_t(c_ta_old), 1. };
                ts::point_ts<ta::generic_dt> pts_new{ gta_t(c_ta_new), 10. };
                // -----
                std::string fn("dtss_save_merge/calendar_new_in_old.db");

                // save initital data
                db.save(fn, pts_old, false);
                auto find_res = db.find("dtss_save_merge/calendar_new_in_old\\.db");
                FAST_CHECK_EQ(find_res.size(), 1);

                // add data to the same path
                db.save(fn, pts_new, false);

                // check merged data
                ts::point_ts<ta::generic_dt> res = db.read("dtss_save_merge/calendar_new_in_old.db", core::utcperiod{ });
                // time-axis
                FAST_CHECK_EQ(res.ta.gt, time_axis::generic_dt::CALENDAR);
                FAST_CHECK_EQ(res.total_period().start, c_ta_old.total_period().start);
                FAST_CHECK_EQ(res.total_period().end, c_ta_old.total_period().end);
                // values
                FAST_CHECK_EQ(res.v.at(0), 1.);
                FAST_CHECK_EQ(res.v.at(drop - 1), 1.);
                FAST_CHECK_EQ(res.v.at(drop), 10.);
                FAST_CHECK_EQ(res.v.at(n - drop - 1), 10.);
                FAST_CHECK_EQ(res.v.at(n - drop), 1.);
                FAST_CHECK_EQ(res.v.at(n - 1), 1.);

                // cleanup
                db.remove(fn);
                find_res = db.find(string("dtss_save_merge/calendar_new_in_old\\.db"));
                FAST_CHECK_EQ(find_res.size(), 0);
            }
            SUBCASE("old contained in new") {
                // data
                const std::size_t extra = 10u;  // points to drop from start/end of old
                // -----
                const core::utctimespan dt = core::calendar::DAY;
                const std::size_t n = 100;
                const core::utctime t0_old = utc_ptr->time(2016, 1, 1);
                const core::utctime t0_new = t0_old - dt*extra;
                // -----
                ta::calendar_dt c_ta_old{ utc_ptr, t0_old, dt, n };
                ta::calendar_dt c_ta_new{ utc_ptr, t0_new, dt, n + extra };
                ts::point_ts<ta::generic_dt> pts_old{ gta_t(c_ta_old), 1. };
                ts::point_ts<ta::generic_dt> pts_new{ gta_t(c_ta_new), 10. };
                // -----
                std::string fn("dtss_save_merge/calendar_old_in_new.db");

                // save initital data
                db.save(fn, pts_old, false);
                auto find_res = db.find("dtss_save_merge/calendar_old_in_new\\.db");
                FAST_CHECK_EQ(find_res.size(), 1);

                // add data to the same path
                db.save(fn, pts_new, false);

                // check merged data
                ts::point_ts<ta::generic_dt> res = db.read("dtss_save_merge/calendar_old_in_new.db", core::utcperiod{ });
                // time-axis
                FAST_CHECK_EQ(res.ta.gt, time_axis::generic_dt::CALENDAR);
                FAST_CHECK_EQ(res.total_period().start, c_ta_new.total_period().start);
                FAST_CHECK_EQ(res.total_period().end, c_ta_new.total_period().end);
                // values
                FAST_CHECK_EQ(res.v.at(0), 10.);
                FAST_CHECK_EQ(res.v.at(n - 1), 10.);

                // cleanup
                db.remove(fn);
                find_res = db.find(string("dtss_save_merge/calendar_old_in_new\\.db"));
                FAST_CHECK_EQ(find_res.size(), 0);
            }
            SUBCASE("new overlap start of old") {
                // data
                const core::utctimespan dt = core::calendar::DAY;
                const std::size_t n = 100;
                const core::utctime t0_old = utc_ptr->time(2016, 1, 1);
                const core::utctime t0_new = t0_old - dt*n/2;
                // -----
                ta::calendar_dt c_ta_old{ utc_ptr, t0_old, dt, n };
                ta::calendar_dt c_ta_new{ utc_ptr, t0_new, dt, n };
                ts::point_ts<ta::generic_dt> pts_old{ gta_t(c_ta_old), 1. };
                ts::point_ts<ta::generic_dt> pts_new{ gta_t(c_ta_new), 10. };
                // -----
                std::string fn("dtss_save_merge/calendar_new_over_start.db");

                FAST_REQUIRE_LT(t0_new, t0_old);

                // save initital data
                db.save(fn, pts_old, false);
                auto find_res = db.find("dtss_save_merge/calendar_new_over_start\\.db");
                FAST_CHECK_EQ(find_res.size(), 1);

                // add data to the same path
                db.save(fn, pts_new, false);

                // check merged data
                ts::point_ts<ta::generic_dt> res = db.read("dtss_save_merge/calendar_new_over_start.db", core::utcperiod{ });
                // time-axis
                FAST_CHECK_EQ(res.ta.gt, time_axis::generic_dt::CALENDAR);
                FAST_CHECK_EQ(res.total_period().start, c_ta_new.total_period().start);
                FAST_CHECK_EQ(res.total_period().end, c_ta_old.total_period().end);
                // values
                FAST_CHECK_EQ(res.v.at(0), 10.);
                FAST_CHECK_EQ(res.v.at(n - 1), 10.);
                FAST_CHECK_EQ(res.v.at(n), 1.);
                FAST_CHECK_EQ(res.v.at(n + n/2 - 1), 1.);

                // cleanup
                db.remove(fn);
                find_res = db.find(string("dtss_save_merge/calendar_new_over_start\\.db"));
                FAST_CHECK_EQ(find_res.size(), 0);
            }
            SUBCASE("new overlap end of old") {
                // data
                const core::utctimespan dt = core::calendar::DAY;
                const std::size_t n = 100;
                const core::utctime t0_old = utc_ptr->time(2016, 1, 1);
                const core::utctime t0_new = t0_old + dt*n/2;
                // -----
                ta::calendar_dt c_ta_old{ utc_ptr, t0_old, dt, n };
                ta::calendar_dt c_ta_new{ utc_ptr, t0_new, dt, n };
                ts::point_ts<ta::generic_dt> pts_old{ gta_t(c_ta_old), 1. };
                ts::point_ts<ta::generic_dt> pts_new{ gta_t(c_ta_new), 10. };
                // -----
                std::string fn("dtss_save_merge/calendar_new_over_end.db");

                // save initital data
                db.save(fn, pts_old, false);
                auto find_res = db.find("dtss_save_merge/calendar_new_over_end\\.db");
                FAST_CHECK_EQ(find_res.size(), 1);

                // add data to the same path
                db.save(fn, pts_new, false);

                // check merged data
                ts::point_ts<ta::generic_dt> res = db.read("dtss_save_merge/calendar_new_over_end.db", core::utcperiod{ });
                // time-axis
                FAST_CHECK_EQ(res.ta.gt, time_axis::generic_dt::CALENDAR);
                FAST_CHECK_EQ(res.total_period().start, c_ta_old.total_period().start);
                FAST_CHECK_EQ(res.total_period().end, c_ta_new.total_period().end);
                // values
                FAST_CHECK_EQ(res.v.at(0), 1.);
                FAST_CHECK_EQ(res.v.at(n/2 - 1), 1.);
                FAST_CHECK_EQ(res.v.at(n/2), 10.);
                FAST_CHECK_EQ(res.v.at(n + n/2 - 1), 10.);

                // cleanup
                db.remove(fn);
                find_res = db.find(string("dtss_save_merge/calendar_new_over_end\\.db"));
                FAST_CHECK_EQ(find_res.size(), 0);
            }
            SUBCASE("consecutive without gap") {
                // data
                const core::utctimespan dt = core::calendar::DAY;
                const std::size_t n = 100;
                const core::utctime t0_old = utc_ptr->time(2016, 1, 1);
                const core::utctime t0_new = t0_old + dt*n;
                // -----
                ta::calendar_dt c_ta_old{ utc_ptr, t0_old, dt, n };
                ta::calendar_dt c_ta_new{ utc_ptr, t0_new, dt, n };
                ts::point_ts<ta::generic_dt> pts_old{ gta_t(c_ta_old), 1. };
                ts::point_ts<ta::generic_dt> pts_new{ gta_t(c_ta_new), 10. };
                // -----
                std::string fn("dtss_save_merge/calendar_consec.db");

                // save initital data
                db.save(fn, pts_old, false);
                auto find_res = db.find("dtss_save_merge/calendar_consec\\.db");
                FAST_CHECK_EQ(find_res.size(), 1);

                // add data to the same path
                db.save(fn, pts_new, false);

                // check merged data
                ts::point_ts<ta::generic_dt> res = db.read("dtss_save_merge/calendar_consec.db", core::utcperiod{ });
                // time-axis
                FAST_CHECK_EQ(res.ta.gt, time_axis::generic_dt::CALENDAR);
                FAST_CHECK_EQ(res.total_period().start, c_ta_old.total_period().start);
                FAST_CHECK_EQ(res.total_period().end, c_ta_new.total_period().end);
                // values
                FAST_CHECK_EQ(res.v.at(0), 1.);
                FAST_CHECK_EQ(res.v.at(n - 1), 1.);
                FAST_CHECK_EQ(res.v.at(n), 10.);
                FAST_CHECK_EQ(res.v.at(2 * n - 1), 10.);

                // cleanup
                db.remove(fn);
                find_res = db.find(string("dtss_save_merge/calendar_consec\\.db"));
                FAST_CHECK_EQ(find_res.size(), 0);
            }
            SUBCASE("new after old with gap") {
                // data
                const core::utctimespan dt = core::calendar::DAY;
                const std::size_t n = 100;
                const core::utctime t0_old = utc_ptr->time(2016, 1, 1);
                const core::utctime t0_new = t0_old + 2*n*dt;
                // -----
                ta::calendar_dt c_ta_old{ utc_ptr, t0_old, dt, n };
                ta::calendar_dt c_ta_new{ utc_ptr, t0_new, dt, n };
                ts::point_ts<ta::generic_dt> pts_old{ gta_t(c_ta_old), 1. };
                ts::point_ts<ta::generic_dt> pts_new{ gta_t(c_ta_new), 10. };
                // -----
                std::string fn("dtss_save_merge/calendar_gap_after.db");

                // save initital data
                db.save(fn, pts_old, false);
                auto find_res = db.find("dtss_save_merge/calendar_gap_after\\.db");
                FAST_CHECK_EQ(find_res.size(), 1);

                // add data to the same path
                db.save(fn, pts_new, false);

                // check merged data
                ts::point_ts<ta::generic_dt> res = db.read("dtss_save_merge/calendar_gap_after.db", core::utcperiod{ });
                // time-axis
                FAST_CHECK_EQ(res.ta.gt, time_axis::generic_dt::CALENDAR);
                FAST_CHECK_EQ(res.ta.size(), 3*n);
                FAST_CHECK_EQ(res.total_period().start, c_ta_old.total_period().start);
                FAST_CHECK_EQ(res.total_period().end, c_ta_new.total_period().end);
                // values
                FAST_CHECK_EQ(res.v.at(0), 1.);
                FAST_CHECK_EQ(res.v.at(n - 1), 1.);
                FAST_CHECK_UNARY(std::isnan(res.v.at(n)));
                FAST_CHECK_UNARY(std::isnan(res.v.at(2*n - 1)));
                FAST_CHECK_EQ(res.v.at(2 * n), 10.);
                FAST_CHECK_EQ(res.v.at(3 * n - 1), 10.);

                // cleanup
                db.remove(fn);
                find_res = db.find(string("dtss_save_merge/calendar_gap_after\\.db"));
                FAST_CHECK_EQ(find_res.size(), 0);
            }
            SUBCASE("new before old with gap") {
                // data
                const core::utctimespan dt = core::calendar::DAY;
                const std::size_t n = 100;
                const core::utctime t0_old = utc_ptr->time(2016, 1, 1);
                const core::utctime t0_new = t0_old - 2*n*dt;
                // -----
                ta::calendar_dt c_ta_old{ utc_ptr, t0_old, dt, n };
                ta::calendar_dt c_ta_new{ utc_ptr, t0_new, dt, n };
                ts::point_ts<ta::generic_dt> pts_old{ gta_t(c_ta_old), 1. };
                ts::point_ts<ta::generic_dt> pts_new{ gta_t(c_ta_new), 10. };
                // -----
                std::string fn("dtss_save_merge/calendar_gap_before.db");

                // save initital data
                db.save(fn, pts_old, false);
                auto find_res = db.find("dtss_save_merge/calendar_gap_before\\.db");
                FAST_CHECK_EQ(find_res.size(), 1);

                // add data to the same path
                db.save(fn, pts_new, false);

                // check merged data
                ts::point_ts<ta::generic_dt> res = db.read("dtss_save_merge/calendar_gap_before.db", core::utcperiod{ });
                // time-axis
                FAST_CHECK_EQ(res.ta.gt, time_axis::generic_dt::CALENDAR);
                FAST_CHECK_EQ(res.ta.size(), 3*n);
                FAST_CHECK_EQ(res.total_period().start, c_ta_new.total_period().start);
                FAST_CHECK_EQ(res.total_period().end, c_ta_old.total_period().end);
                // values
                FAST_CHECK_EQ(res.v.at(0), 10.);
                FAST_CHECK_EQ(res.v.at(n - 1), 10.);
                FAST_CHECK_UNARY(std::isnan(res.v.at(n)));
                FAST_CHECK_UNARY(std::isnan(res.v.at(2*n - 1)));
                FAST_CHECK_EQ(res.v.at(2 * n), 1.);
                FAST_CHECK_EQ(res.v.at(3 * n - 1), 1.);

                // cleanup
                db.remove(fn);
                find_res = db.find(string("dtss_save_merge/calendar_gap_before\\.db"));
                FAST_CHECK_EQ(find_res.size(), 0);
            }
        }
        SUBCASE("point_dt") {
            // data
            const core::utctimespan dt = core::calendar::DAY;
            const std::size_t n = 100;
            const core::utctime t0_old = utc_ptr->time(2016, 1, 1);
            const core::utctimespan offset = core::deltaminutes(25);  // to misalign old and new timepoints
            // -----
            std::vector<core::utctime> old_timepoints{};
            const std::vector<double> old_values(n, 1.);
            // -----
            std::vector<core::utctime> new_timepoints{};
            std::vector<double> new_values(n, 10.);
            // -----
            old_timepoints.reserve(n + 1);
            for ( std::size_t i = 0; i <= n; ++i ) {
                old_timepoints.emplace_back(t0_old + dt*i);
            }

            SUBCASE("exact") {
                // data
                ta::point_dt p_ta{ old_timepoints };
                ts::point_ts<ta::generic_dt> pts_old{ gta_t(p_ta), old_values };
                ts::point_ts<ta::generic_dt> pts_new{ gta_t(p_ta), new_values };
                // -----
                std::string fn("dtss_save_merge/point_exact.db");

                // save initital data
                db.save(fn, pts_old, false);
                auto find_res = db.find("dtss_save_merge/point_exact\\.db");
                FAST_CHECK_EQ(find_res.size(), 1);

                // add data to the same path
                db.save(fn, pts_new, false);

                // check merged data
                ts::point_ts<ta::generic_dt> res = db.read("dtss_save_merge/point_exact.db", core::utcperiod{ });
                // time-axis
                FAST_CHECK_EQ(res.ta.gt, time_axis::generic_dt::POINT);
                FAST_CHECK_EQ(res.total_period().start, p_ta.total_period().start);
                FAST_CHECK_EQ(res.total_period().end, p_ta.total_period().end);
                // values
                FAST_CHECK_EQ(res.v.at(0), 10.);
                FAST_CHECK_EQ(res.v.at(n - 1), 10.);

                // cleanup
                db.remove(fn);
                find_res = db.find(string("dtss_save_merge/point_exact\\.db"));
                FAST_CHECK_EQ(find_res.size(), 0);
            }
            SUBCASE("new contained in old") {
                // data
                const std::size_t drop = 10u;  // points to drop from start/end of old
                const core::utctime t0_new = t0_old + offset + dt*drop;
                // -----
                new_timepoints.reserve(n - 2 * drop);
                for ( std::size_t i = 0; i <= n - 2 * drop; ++i ) {
                    new_timepoints.emplace_back(t0_new + i * dt);
                }
                new_values.resize(new_timepoints.size() - 1);
                // -----
                ta::point_dt p_ta_old{ old_timepoints };
                ta::point_dt p_ta_new{ new_timepoints };
                ts::point_ts<ta::generic_dt> pts_old{ gta_t(p_ta_old), old_values };
                ts::point_ts<ta::generic_dt> pts_new{ gta_t(p_ta_new), new_values };
                // -----
                std::string fn("dtss_save_merge/point_new_in_old.db");

                // save initital data
                db.save(fn, pts_old, false);
                auto find_res = db.find("dtss_save_merge/point_new_in_old\\.db");
                FAST_CHECK_EQ(find_res.size(), 1);

                // add data to the same path
                db.save(fn, pts_new, false);

                // check merged data
                ts::point_ts<ta::generic_dt> res = db.read("dtss_save_merge/point_new_in_old.db", core::utcperiod{ });
                // time-axis
                FAST_CHECK_EQ(res.ta.gt, time_axis::generic_dt::POINT);
                FAST_CHECK_EQ(res.total_period().start, p_ta_old.total_period().start);
                FAST_CHECK_EQ(res.total_period().end, p_ta_old.total_period().end);
                // values
                //  - NB! (n+1) because the point_dt's are unaligned, so a new point is introduced
                FAST_CHECK_EQ(res.v.at(0), 1.);
                FAST_CHECK_EQ(res.v.at(drop), 1.);  // this is the extra new point
                FAST_CHECK_EQ(res.v.at(drop + 1), 10.);
                FAST_CHECK_EQ(res.v.at((n+1) - drop - 1), 10.);
                FAST_CHECK_EQ(res.v.at((n+1) - drop), 1.);
                FAST_CHECK_EQ(res.v.at((n+1) - 1), 1.);

                // cleanup
                db.remove(fn);
                find_res = db.find(string("dtss_save_merge/point_new_in_old\\.db"));
                FAST_CHECK_EQ(find_res.size(), 0);
            }
            SUBCASE("old contained in new") {
                // data
                const std::size_t extra = 10u;  // points to drop from start/end of old
                const core::utctime t0_new = t0_old + offset - dt*extra;
                // -----
                new_timepoints.reserve(n + 2 * extra);
                for ( std::size_t i = 0; i <= n + 2 * extra; ++i ) {
                    new_timepoints.emplace_back(t0_new + i * dt);
                }
                new_values.resize(new_timepoints.size() - 1, 10.);
                // -----
                ta::point_dt p_ta_old{ old_timepoints };
                ta::point_dt p_ta_new{ new_timepoints };
                ts::point_ts<ta::generic_dt> pts_old{ gta_t(p_ta_old), old_values };
                ts::point_ts<ta::generic_dt> pts_new{ gta_t(p_ta_new), new_values };
                // -----
                std::string fn("dtss_save_merge/point_old_in_new.db");

                // save initital data
                db.save(fn, pts_old, false);
                auto find_res = db.find("dtss_save_merge/point_old_in_new\\.db");
                FAST_CHECK_EQ(find_res.size(), 1);

                // add data to the same path
                db.save(fn, pts_new, false);

                // check merged data
                ts::point_ts<ta::generic_dt> res = db.read("dtss_save_merge/point_old_in_new.db", core::utcperiod{ });
                // time-axis
                FAST_CHECK_EQ(res.ta.gt, time_axis::generic_dt::POINT);
                FAST_CHECK_EQ(res.total_period().start, p_ta_new.total_period().start);
                FAST_CHECK_EQ(res.total_period().end, p_ta_new.total_period().end);
                // values
                FAST_CHECK_EQ(res.v.at(0), 10.);
                FAST_CHECK_EQ(res.v.at(n - 1), 10.);

                // cleanup
                db.remove(fn);
                find_res = db.find(string("dtss_save_merge/point_old_in_new\\.db"));
                FAST_CHECK_EQ(find_res.size(), 0);
            }
            SUBCASE("new overlap start of old") {
                // data
                const core::utctime t0_new = t0_old + offset - dt*n/2;
                for ( std::size_t i = 0; i <= n; ++i ) {
                    new_timepoints.emplace_back(t0_new + i * dt);
                }
                // -----
                ta::point_dt p_ta_old{ old_timepoints };
                ta::point_dt p_ta_new{ new_timepoints };
                ts::point_ts<ta::generic_dt> pts_old{ gta_t(p_ta_old), old_values };
                ts::point_ts<ta::generic_dt> pts_new{ gta_t(p_ta_new), new_values };
                // -----
                std::string fn("dtss_save_merge/point_new_over_start.db");

                // save initital data
                db.save(fn, pts_old, false);
                auto find_res = db.find("dtss_save_merge/point_new_over_start\\.db");
                FAST_CHECK_EQ(find_res.size(), 1);

                // add data to the same path
                db.save(fn, pts_new, false);

                // check merged data
                ts::point_ts<ta::generic_dt> res = db.read("dtss_save_merge/point_new_over_start.db", core::utcperiod{ });
                // time-axis
                FAST_CHECK_EQ(res.ta.gt, time_axis::generic_dt::POINT);
                FAST_CHECK_EQ(res.total_period().start, p_ta_new.total_period().start);
                FAST_CHECK_EQ(res.total_period().end, p_ta_old.total_period().end);
                // values
                FAST_CHECK_EQ(res.v.at(0), 10.);
                FAST_CHECK_EQ(res.v.at(n - 1), 10.);
                FAST_CHECK_EQ(res.v.at(n), 1.);
                FAST_CHECK_EQ(res.v.at(n + n/2 - 1), 1.);

                // cleanup
                db.remove(fn);
                find_res = db.find(string("dtss_save_merge/point_new_over_start\\.db"));
                FAST_CHECK_EQ(find_res.size(), 0);
            }
            SUBCASE("new overlap end of old") {
                // data
                const core::utctime t0_new = t0_old + offset + dt*n/2;
                for ( std::size_t i = 0; i <= n; ++i ) {
                    new_timepoints.emplace_back(t0_new + i * dt);
                }
                // -----
                ta::point_dt p_ta_old{ old_timepoints };
                ta::point_dt p_ta_new{ new_timepoints };
                ts::point_ts<ta::generic_dt> pts_old{ gta_t(p_ta_old), old_values };
                ts::point_ts<ta::generic_dt> pts_new{ gta_t(p_ta_new), new_values };
                // -----
                std::string fn("dtss_save_merge/point_new_over_end.db");

                // save initital data
                db.save(fn, pts_old, false);
                auto find_res = db.find("dtss_save_merge/point_new_over_end\\.db");
                FAST_CHECK_EQ(find_res.size(), 1);

                // add data to the same path
                db.save(fn, pts_new, false);

                // check merged data
                ts::point_ts<ta::generic_dt> res = db.read("dtss_save_merge/point_new_over_end.db", core::utcperiod{ });
                // time-axis
                FAST_CHECK_EQ(res.ta.gt, time_axis::generic_dt::POINT);
                FAST_CHECK_EQ(res.total_period().start, p_ta_old.total_period().start);
                FAST_CHECK_EQ(res.total_period().end, p_ta_new.total_period().end);
                // values
                FAST_CHECK_EQ(res.v.at(0), 1.);
                FAST_CHECK_EQ(res.v.at(n/2), 1.);
                FAST_CHECK_EQ(res.v.at(n/2 + 1), 10.);
                FAST_CHECK_EQ(res.v.at(n + n/2), 10.);

                // cleanup
                db.remove(fn);
                find_res = db.find(string("dtss_save_merge/point_new_over_end\\.db"));
                FAST_CHECK_EQ(find_res.size(), 0);
            }
            SUBCASE("consecutive without gap") {
                // data
                const core::utctime t0_new = t0_old + n * dt;
                for ( std::size_t i = 0; i <= n; ++i ) {
                    new_timepoints.emplace_back(t0_new + i * dt);
                }
                // -----
                ta::point_dt p_ta_old{ old_timepoints };
                ta::point_dt p_ta_new{ new_timepoints };
                ts::point_ts<ta::generic_dt> pts_old{ gta_t(p_ta_old), old_values };
                ts::point_ts<ta::generic_dt> pts_new{ gta_t(p_ta_new), new_values };
                // -----
                std::string fn("dtss_save_merge/point_consec.db");

                // save initital data
                db.save(fn, pts_old, false);
                auto find_res = db.find("dtss_save_merge/point_consec\\.db");
                FAST_CHECK_EQ(find_res.size(), 1);

                // add data to the same path
                db.save(fn, pts_new, false);

                // check merged data
                ts::point_ts<ta::generic_dt> res = db.read("dtss_save_merge/point_consec.db", core::utcperiod{ });
                // time-axis
                FAST_CHECK_EQ(res.ta.gt, time_axis::generic_dt::POINT);
                FAST_CHECK_EQ(res.total_period().start, p_ta_old.total_period().start);
                FAST_CHECK_EQ(res.total_period().end, p_ta_new.total_period().end);
                // values
                FAST_CHECK_EQ(res.v.at(0), 1.);
                FAST_CHECK_EQ(res.v.at(n - 1), 1.);
                FAST_CHECK_EQ(res.v.at(n), 10.);
                FAST_CHECK_EQ(res.v.at(2 * n - 1), 10.);

                // cleanup
                db.remove(fn);
                find_res = db.find(string("dtss_save_merge/point_consec\\.db"));
                FAST_CHECK_EQ(find_res.size(), 0);
            }
            SUBCASE("new after old with gap") {
                // data
                const core::utctime t0_new = t0_old + n * dt + offset;
                for ( std::size_t i = 0; i <= n; ++i ) {
                    new_timepoints.emplace_back(t0_new + i * dt);
                }
                // -----
                ta::point_dt p_ta_old{ old_timepoints };
                ta::point_dt p_ta_new{ new_timepoints };
                ts::point_ts<ta::generic_dt> pts_old{ gta_t(p_ta_old), old_values };
                ts::point_ts<ta::generic_dt> pts_new{ gta_t(p_ta_new), new_values };
                // -----
                std::string fn("dtss_save_merge/point_gap_after.db");

                // save initital data
                db.save(fn, pts_old, false);
                auto find_res = db.find("dtss_save_merge/point_gap_after\\.db");
                FAST_CHECK_EQ(find_res.size(), 1);

                // add data to the same path
                db.save(fn, pts_new, false);

                // check merged data
                ts::point_ts<ta::generic_dt> res = db.read("dtss_save_merge/point_gap_after.db", core::utcperiod{ });
                // time-axis
                FAST_CHECK_EQ(res.ta.gt, time_axis::generic_dt::POINT);
                FAST_CHECK_EQ(res.ta.size(), 2 * n + 1);
                FAST_CHECK_EQ(res.total_period().start, p_ta_old.total_period().start);
                FAST_CHECK_EQ(res.total_period().end, p_ta_new.total_period().end);
                // values
                FAST_CHECK_EQ(res.v.at(0), 1.);
                FAST_CHECK_EQ(res.v.at(n - 1), 1.);
                FAST_CHECK_UNARY(std::isnan(res.v.at(n)));
                FAST_CHECK_EQ(res.v.at(n + 1), 10.);
                FAST_CHECK_EQ(res.v.at(2 * n), 10.);

                // cleanup
                db.remove(fn);
                find_res = db.find(string("dtss_save_merge/point_gap_after\\.db"));
                FAST_CHECK_EQ(find_res.size(), 0);
            }
            SUBCASE("new before old with gap") {
                // data
                const core::utctime t0_new = t0_old - n * dt - offset;
                for ( std::size_t i = 0; i <= n; ++i ) {
                    new_timepoints.emplace_back(t0_new + i * dt);
                }
                // -----
                ta::point_dt p_ta_old{ old_timepoints };
                ta::point_dt p_ta_new{ new_timepoints };
                ts::point_ts<ta::generic_dt> pts_old{ gta_t(p_ta_old), old_values };
                ts::point_ts<ta::generic_dt> pts_new{ gta_t(p_ta_new), new_values };
                // -----
                std::string fn("dtss_save_merge/point_gap_before.db");

                // save initital data
                db.save(fn, pts_old, false);
                auto find_res = db.find("dtss_save_merge/point_gap_before\\.db");
                FAST_CHECK_EQ(find_res.size(), 1);

                // add data to the same path
                db.save(fn, pts_new, false);

                // check merged data
                ts::point_ts<ta::generic_dt> res = db.read("dtss_save_merge/point_gap_before.db", core::utcperiod{ });
                // time-axis
                FAST_CHECK_EQ(res.ta.gt, time_axis::generic_dt::POINT);
                FAST_CHECK_EQ(res.ta.size(), 2 * n + 1);
                FAST_CHECK_EQ(res.total_period().start, p_ta_new.total_period().start);
                FAST_CHECK_EQ(res.total_period().end, p_ta_old.total_period().end);
                // values
                FAST_CHECK_EQ(res.v.at(0), 10.);
                FAST_CHECK_EQ(res.v.at(n - 1), 10.);
                FAST_CHECK_UNARY(std::isnan(res.v.at(n)));
                FAST_CHECK_EQ(res.v.at(n + 1), 1.);
                FAST_CHECK_EQ(res.v.at(2 * n), 1.);

                // cleanup
                db.remove(fn);
                find_res = db.find(string("dtss_save_merge/point_gap_before\\.db"));
                FAST_CHECK_EQ(find_res.size(), 0);
            }
        }
        SUBCASE("force overwrite") {
            // data
            const std::size_t drop = 10u;  // points to drop from start/end of old
            // -----
            const core::utctimespan dt = core::calendar::DAY;
            const std::size_t n = 100;
            const core::utctime t0_old = utc_ptr->time(2016, 1, 1);
            const core::utctime t0_new = t0_old + dt*drop;
            // -----
            ta::fixed_dt f_ta_old{ t0_old, dt, n };
            ta::fixed_dt f_ta_new{ t0_new, dt, n - 2*drop };
            ts::point_ts<ta::generic_dt> pts_old{ gta_t(f_ta_old), 1. };
            ts::point_ts<ta::generic_dt> pts_new{ gta_t(f_ta_new), 10. };
            // -----
            std::string fn("dtss_save_merge/force_overwrite.db");

            // save initital data
            db.save(fn, pts_old, false);
            auto find_res = db.find("dtss_save_merge/force_overwrite\\.db");
            FAST_CHECK_EQ(find_res.size(), 1);

            // add data to the same path with overwrite
            db.save(fn, pts_new, true);

            // check merged data
            ts::point_ts<ta::generic_dt> res = db.read("dtss_save_merge/force_overwrite.db", core::utcperiod{ });
            // time-axis
            FAST_CHECK_EQ(res.ta.gt, time_axis::generic_dt::FIXED);
            FAST_CHECK_EQ(res.total_period().start, f_ta_new.total_period().start);
            FAST_CHECK_EQ(res.total_period().end, f_ta_new.total_period().end);
            // values
            FAST_CHECK_EQ(res.v.at(0), 10.);
            FAST_CHECK_EQ(res.v.at(n - 2 * drop - 1), 10.);

            // cleanup
            db.remove(fn);
            find_res = db.find(string("dtss_save_merge/force_overwrite\\.db"));
            FAST_CHECK_EQ(find_res.size(), 0);
        }
    }
}

TEST_CASE("dtss_baseline") {
    using namespace shyft::dtss;
    using namespace shyft::time_series::dd;
    using time_series::point_ts;
    using time_series::ts_point_fx;
    using std::cout;
    auto utc=make_shared<calendar>();
    auto t = utc->time(2016, 1, 1);
    auto dt = deltahours(1);
    const int n = 24 * 365 * 5/3;//24*365*5;

    vector<point_ts<time_axis::fixed_dt>> ftsv;
    const size_t n_ts=10*83;
    arma::mat a_mat(n,n_ts);
    time_axis::fixed_dt fta(t, dt, n);
    //time_axis::generic_dt gta{t,dt*24,size_t(n/24)};
    const auto stair_case=ts_point_fx::POINT_AVERAGE_VALUE;
    ts_vector_t tsv;
    for(size_t i=0;i<n_ts;++i) {
        tsv.emplace_back(to_string(i),apoint_ts(fta,i*10.0,stair_case));
        ftsv.emplace_back(fta,i*10.0,stair_case);
        for(size_t t =0;t<n;++t)
            a_mat(t,i) = i*10.0;
    }
    tsv = 3.0*tsv;


    //-- establish benchmark core-ts
    auto t0 = timing::now();

    vector<vector<double>> bmr;bmr.reserve(n_ts);
    for(const auto &ts:ftsv) {
        auto calc= 3.0* ts;
        vector<double> r;r.reserve(calc.size());
        for(size_t i=0;i<calc.size();++i)
            r.emplace_back(calc.value(i));
        bmr.emplace_back(move(r));
    }
    auto t1 =timing::now();

    //-- establish benchmark armadillo
    vector<vector<double>> amr;amr.reserve(n_ts);
    auto a_res= (a_mat*3.0).eval();
    for(size_t i=0;i<n_ts;++i) {
        amr.emplace_back(arma::conv_to<vector<double>>::from(a_res.col(i)) );
    }
    auto t2 = timing::now();

    //-- establish timing for apoint_ts eval.
    vector<vector<double>> xmr;xmr.reserve(n_ts);
    //auto xtsv = deflate_ts_vector<point_ts<time_axis::generic_dt>>(tsv);
    for(const auto &ts:tsv) {
        xmr.emplace_back(move(ts.values()));
    }
    auto t3 = timing::now();

    FAST_CHECK_EQ(bmr.size(),n_ts);
    FAST_CHECK_EQ(amr.size(),n_ts);
    FAST_CHECK_EQ(xmr.size(),n_ts);

    cout<<"core-ts base-line n_ts= "<<n_ts<<", n="<<n<<", time="<<double(elapsed_us(t0,t1))/1000.0<<"ms ->"
    << double(n*n_ts)/(elapsed_us(t0,t1)/1e6)/1e6<<" mops/s \n";

    cout<<"api -ts base-line n_ts= "<<n_ts<<", n="<<n<<", time="<<double(elapsed_us(t2,t3))/1000.0<<"ms ->"
    << double(n*n_ts)/(elapsed_us(t2,t3)/1e6)/1e6<<" mops/s \n";

    cout<<"armavec base-line n_ts= "<<n_ts<<", n="<<n<<", time="<<double(elapsed_us(t1,t2))/1000.0<<"ms ->"
    << double(n*n_ts)/(elapsed_us(t1,t2)/1e6)/1e6<<" mops/s \n";

}

TEST_CASE("dtss_ltm") {
    // this is basically just for performance study of
    // for api type of ts-expressions,
    using namespace shyft::dtss;
    using namespace shyft::time_series::dd;
    using shyft::time_series::point_ts;
    using time_series::ts_point_fx;
    using std::cout;
    auto utc=make_shared<calendar>();
    auto t = utc->time(2016, 1, 1);
    auto dt = deltahours(1);
    const int n = 24 * 365 * 5/3;//24*365*5;

    const size_t n_scn=83;
    const size_t n_obj =2;
    const size_t n_ts=n_obj*2*n_scn;

    vector<point_ts<time_axis::fixed_dt>> ftsv;
    arma::mat a_mat(n,n_ts);
    time_axis::fixed_dt fta(t, dt, n);
    time_axis::generic_dt gta{t,dt*24,size_t(n/24)};
    const auto stair_case=ts_point_fx::POINT_AVERAGE_VALUE;
    map<string,apoint_ts> rtsv;
    ts_vector_t stsv;
    for(size_t i=0;i<n_ts;++i) {
        rtsv[to_string(i)] = apoint_ts(fta,i*10.0,stair_case);
        stsv.emplace_back(to_string(i));
        ftsv.emplace_back(fta,i*10.0,stair_case);
        for(size_t t =0;t<n;++t)
            a_mat(t,i) = i*10.0;
    }
    ts_vector_t tsv;
    for(size_t i =0; i<n_scn;++i) {
        apoint_ts sum;
        for(size_t j=0;j<n_obj;++j) {
            size_t p = i*(2*n_obj) + (2*j);
            size_t c = p+1;
            apoint_ts eff=3.0*(stsv[p]-stsv[c]);
            if(j==0)
                sum = eff;
            else
                sum = sum + eff;
        }
        tsv.emplace_back(sum);
    }
    tsv = 1000.0*(tsv.average(gta));


    //-- establish compute binding time
    auto t0 = timing::now();
    size_t bind_count{0};
    for(auto&sts:tsv) {
        auto ts_refs=sts.find_ts_bind_info();
        for(auto& bi:ts_refs) {
            bi.ts.bind(rtsv[bi.reference]);
            bind_count++;
        }
    }
    for(auto&sts:tsv)
        sts.do_bind();

    auto t1 =timing::now();
    //-- establish timing for apoint_ts eval.
    auto xmr = deflate_ts_vector<point_ts<time_axis::generic_dt>>(tsv);
    auto t2 = timing::now();

    FAST_CHECK_EQ(xmr.size(),n_scn);
    FAST_CHECK_EQ(bind_count,n_ts);
    cout<<"bind phase n_ts= "<<n_ts<<", n="<<n<<", time="<<double(elapsed_us(t0,t1))/1000.0<<"ms\n";
    cout<<"eval phase n_ts= "<<n_ts<<", n="<<n<<", time="<<double(elapsed_us(t1,t2))/1000.0<<"ms ->"
    << double(n*n_ts*(2+1))/(elapsed_us(t1,t2)/1e6)/1e6<<" mops/s \n";

}

TEST_CASE("dtss_container_wrapping") {
    auto tmpdir = (fs::temp_directory_path()/fs::unique_path());

    shyft::core::calendar utc;

    using shyft::time_axis::generic_dt;
    using shyft::time_series::point_ts;
    using shyft::time_series::ts_point_fx;
    // -----
    using shyft::dtss::ts_db;
    using cwrp_t = shyft::dtss::container_wrapper<ts_db>;
    
    SUBCASE("dispatch save through container") {
        std::string ts_name{ "test" };

        generic_dt ta{ utc.time(2002, 2, 2), calendar::HOUR, 24 };
        point_ts<generic_dt> ts{ ta, 15., shyft::time_series::ts_point_fx::POINT_INSTANT_VALUE };

        cwrp_t container{ std::make_unique<ts_db>(tmpdir.string()) };

        container.save(ts_name, ts);
        FAST_REQUIRE_UNARY( fs::is_regular_file(tmpdir/ts_name) );
    }

    SUBCASE("dispatch read through container") {
        std::string ts_name{ "test" };

        generic_dt ta{ utc.time(2002, 2, 2), calendar::HOUR, 24 };
        point_ts<generic_dt> ts{ ta, 15., shyft::time_series::ts_point_fx::POINT_INSTANT_VALUE };

        cwrp_t container{ std::make_unique<ts_db>(tmpdir.string()) };
        
        container.save(ts_name, ts);
        FAST_CHECK_UNARY( fs::is_regular_file(tmpdir/ts_name) );

        auto ts_new = container.read(ts_name, ta.total_period());
        TS_ASSERT_EQUALS( ts_new, ts );
    }

    SUBCASE("dispatch remove through container") {
        std::string ts_name{ "test" };

        generic_dt ta{ utc.time(2002, 2, 2), calendar::HOUR, 24 };
        point_ts<generic_dt> ts{ ta, 15., shyft::time_series::ts_point_fx::POINT_INSTANT_VALUE };

        cwrp_t container{ std::make_unique<ts_db>(tmpdir.string()) };

        container.save(ts_name, ts);
        FAST_CHECK_UNARY( fs::is_regular_file(tmpdir/ts_name) );

        container.remove(ts_name);
        FAST_REQUIRE_UNARY_FALSE( fs::exists(tmpdir/ts_name) );
    }

    SUBCASE("dispatch get_ts_info through container") {
        std::string ts_name{ "test" };

        generic_dt ta{ utc.time(2002, 2, 2), calendar::HOUR, 24 };
        point_ts<generic_dt> ts{ ta, 15., shyft::time_series::ts_point_fx::POINT_INSTANT_VALUE };

        cwrp_t container{ std::make_unique<ts_db>(tmpdir.string()) };

        container.save(ts_name, ts);
        FAST_CHECK_UNARY( fs::is_regular_file(tmpdir/ts_name) );

        auto info = container.get_ts_info(ts_name);
        FAST_REQUIRE_EQ( info.name, ts_name );
    }

    SUBCASE("dispatch find through container") {
        std::string ts_name{ "test" };

        generic_dt ta{ utc.time(2002, 2, 2), calendar::HOUR, 24 };
        point_ts<generic_dt> ts{ ta, 15., shyft::time_series::ts_point_fx::POINT_INSTANT_VALUE };

        cwrp_t container{ std::make_unique<ts_db>(tmpdir.string()) };

        container.save(ts_name, ts);
        FAST_CHECK_UNARY( fs::is_regular_file(tmpdir/ts_name) );

        auto info = container.find(ts_name);
        FAST_REQUIRE_EQ( info.size(), 1 );
        FAST_REQUIRE_EQ( info[0].name, ts_name );
    }
}


namespace {

struct query_test_dtss_container {

    using ts_info = shyft::dtss::ts_info;
    using gta_t = shyft::time_axis::generic_dt;
    using gts_t = shyft::time_series::point_ts<gta_t>;
    using queries_t = std::map<std::string, std::string>;

    std::string root;

    query_test_dtss_container() = default;
    ~query_test_dtss_container() = default;

    query_test_dtss_container(const std::string & root_dir) : root{ root_dir } {}

    query_test_dtss_container(const query_test_dtss_container &) = default;
    query_test_dtss_container & operator=(const query_test_dtss_container &) = default;

    query_test_dtss_container(query_test_dtss_container &&) = default;
    query_test_dtss_container & operator=(query_test_dtss_container &&) = default;

    /*  Container API
     * =============== */

    void save(const std::string & fn, const gts_t & ts, bool overwrite = true, const queries_t & queries = queries_t{}) const {

        FAST_CHECK_UNARY( fn.find("?") == fn.npos );  // there should be no query part in the filename

        auto q = queries.find("my_query");
        FAST_REQUIRE_NE( q, queries.cend() );
        FAST_CHECK_EQ( q->first, std::string{"my_query"} );
        FAST_CHECK_EQ( q->second, std::string{"some_value"} );

        FAST_CHECK_EQ( queries.find("removed"), queries.cend() );
    }

    gts_t read(const std::string & fn, core::utcperiod p, const queries_t & queries = queries_t{}) const {

        FAST_CHECK_UNARY( fn.find("?") == fn.npos );  // there should be no query part in the filename

        auto q = queries.find("my_query");
        FAST_REQUIRE_NE( q, queries.cend() );
        FAST_CHECK_EQ( q->first, std::string{"my_query"} );
        FAST_CHECK_EQ( q->second, std::string{"some_value"} );

        FAST_CHECK_EQ( queries.find("removed"), queries.cend() );
        return gts_t{};
    }

    void remove(const std::string & fn, const queries_t & queries = queries_t{}) const {

        FAST_CHECK_UNARY( fn.find("?") == fn.npos );  // there should be no query part in the filename

        auto q = queries.find("my_query");
        FAST_REQUIRE_NE( q, queries.cend() );
        FAST_CHECK_EQ( q->first, std::string{"my_query"} );
        FAST_CHECK_EQ( q->second, std::string{"some_value"} );

        FAST_CHECK_EQ( queries.find("removed"), queries.cend() );
    }

    ts_info get_ts_info(const std::string & fn, const queries_t & queries = queries_t{}) const {

        FAST_CHECK_UNARY( fn.find("?") == fn.npos );  // there should be no query part in the filename

        auto q = queries.find("my_query");
        FAST_REQUIRE_NE( q, queries.cend() );
        FAST_CHECK_EQ( q->first, std::string{"my_query"} );
        FAST_CHECK_EQ( q->second, std::string{"some_value"} );

        FAST_CHECK_EQ( queries.find("removed"), queries.cend() );

        return ts_info{};
    }

    std::vector<ts_info> find(const std::string & match, const queries_t & queries = queries_t{}) const {
        auto q = queries.find("my_query");
        FAST_REQUIRE_NE( q, queries.cend() );
        FAST_CHECK_EQ( q->first, std::string{"my_query"} );
        FAST_CHECK_EQ( q->second, std::string{"some_value"} );

        FAST_CHECK_EQ( queries.find("removed"), queries.cend() );

        return std::vector<ts_info>{};
    };
};

struct query_test_dtss_dispatcher {

    using queries_t = std::map<std::string, std::string>;
    using container_wrapper_t = dtss::container_wrapper<query_test_dtss_container>;

    inline static const std::string container_query{ "container" };
    inline static const std::array<std::string, 2> remove_queries{{ "removed", container_query }};

    static void create_container(
        const std::string & container_name,
        const std::string & container_type,
        const std::string & root_path,
        shyft::dtss::server<query_test_dtss_dispatcher> & dtss_server
    ) {
        if ( container_type.empty() || container_type == "test" ) {
            dtss_server.container[std::string{"TEST_"} + container_name] = container_wrapper_t{ std::make_unique<query_test_dtss_container>(root_path) };
        } else {
            throw std::runtime_error{ std::string{"Cannot construct unknown container type: "} + container_type };
        }
    }

    static container_wrapper_t & get_container(
        const std::string & container_name, const std::string & container_query,
        dtss::server<query_test_dtss_dispatcher> & dtss_server
    ) {
        decltype(dtss_server.container)::iterator f;
        if ( container_query.empty() || container_query == "test" ) {
            f = dtss_server.container.find(std::string{"TEST_"} + container_name);
        } else {
            throw std::runtime_error{ std::string{"Cannot construct unknown container type: "} + container_query };
        }

        if( f == std::end(dtss_server.container) )
            throw runtime_error(std::string{"Failed to find shyft container: "} + container_name);

        return f->second;
    }
};

}

TEST_CASE("dtss_server_query_to_containers") {

    shyft::core::calendar utc;

    using shyft::time_axis::generic_dt;
    using shyft::time_series::dd::apoint_ts;
    using shyft::time_series::ts_point_fx;
    using ts_vector_t = shyft::time_series::dd::ats_vector;
    using test_server_t = shyft::dtss::server<query_test_dtss_dispatcher>;
    using queries_t = std::map<std::string, std::string>;

    const int port = 40336;
    const std::string container{"container"};

    test_server_t srv{};
    srv.set_listening_port(port);
    srv.add_container(container, "foo/bar/00");  // same as "first"
    srv.start_async();

    shyft::dtss::client cli{ std::string{"localhost:"} + std::to_string(port) };

    /* NOTE: `remove` and `get_ts_info` are not called from the server...
     */

    SUBCASE("Save using url's with queries") {
        core::utctime t = utc.time(2016, 1, 1);
        core::utctimespan dt = core::deltahours(1);
        size_t n = 10;
        generic_dt ta{ t, dt, n };

        ts_vector_t vec{};
        for( std::size_t i = 0; i < 10; ++i ) {
            vec.emplace_back(
                shyft::dtss::shyft_url(container, to_string(i), queries_t{ {"my_query", "some_value"}, {"removed", "value"} }),
                apoint_ts{ ta, i*10.0, ts_point_fx::POINT_INSTANT_VALUE }
            );
        }
        
        cli.store_ts(vec, false, false);
    }
    
    SUBCASE("Read using url's with queries") {
        utcperiod period{ utc.time(2016, 1, 1), utc.time(2017, 1, 1) };

        ts_vector_t vec{};
        for( std::size_t i = 0; i < 10; ++i ) {
            vec.emplace_back(
                apoint_ts{ shyft::dtss::shyft_url(container, to_string(i), queries_t{ {"my_query", "some_value"}, {"removed", "value"} }) }
            );
        }

        auto res = cli.evaluate(vec, period, false, false);
    }

    SUBCASE("Find using url's with queries") {
        cli.find(shyft::dtss::shyft_url(container, "/path/to/something", queries_t{ {"my_query", "some_value"}, {"removed", "value"} }));
    }
}


namespace {

template < int Idx >
struct multicontainer_test_dtss_container {

    using ts_info = shyft::dtss::ts_info;
    using gta_t = shyft::time_axis::generic_dt;
    using gts_t = shyft::time_series::point_ts<gta_t>;
    using queries_t = std::map<std::string, std::string>;

    std::string root;

    multicontainer_test_dtss_container() = default;
    ~multicontainer_test_dtss_container() = default;

    multicontainer_test_dtss_container(const std::string & root_dir) : root{ root_dir } {}

    multicontainer_test_dtss_container(const multicontainer_test_dtss_container &) = default;
    multicontainer_test_dtss_container & operator=(const multicontainer_test_dtss_container &) = default;

    multicontainer_test_dtss_container(multicontainer_test_dtss_container &&) = default;
    multicontainer_test_dtss_container & operator=(multicontainer_test_dtss_container &&) = default;

    /*  Container API
    * =============== */

    void save(const std::string & fn, const gts_t & ts, bool overwrite = true, const queries_t & queries = queries_t{}) const {

        FAST_CHECK_UNARY( fn.find("?") == fn.npos );  // there should be no query part in the filename

        auto q = queries.find("value");
        FAST_REQUIRE_NE( q, queries.cend() );
        FAST_CHECK_EQ( q->first, std::string{"value"} );
        FAST_CHECK_EQ( q->second, std::to_string(Idx) );
    }

    gts_t read(const std::string & fn, core::utcperiod p, const queries_t & queries = queries_t{}) const {

        FAST_CHECK_UNARY( fn.find("?") == fn.npos );  // there should be no query part in the filename

        auto q = queries.find("value");
        FAST_REQUIRE_NE( q, queries.cend() );
        FAST_CHECK_EQ( q->first, std::string{"value"} );
        FAST_CHECK_EQ( q->second, std::to_string(Idx) );
        return gts_t{};
    }

    void remove(const std::string & fn, const queries_t & queries = queries_t{}) const {

        FAST_CHECK_UNARY( fn.find("?") == fn.npos );  // there should be no query part in the filename

        auto q = queries.find("value");
        FAST_REQUIRE_NE( q, queries.cend() );
        FAST_CHECK_EQ( q->first, std::string{"value"} );
        FAST_CHECK_EQ( q->second, std::to_string(Idx) );
    }

    ts_info get_ts_info(const std::string & fn, const queries_t & queries = queries_t{}) const {

        FAST_CHECK_UNARY( fn.find("?") == fn.npos );  // there should be no query part in the filename

        auto q = queries.find("value");
        FAST_REQUIRE_NE( q, queries.cend() );
        FAST_CHECK_EQ( q->first, std::string{"value"} );
        FAST_CHECK_EQ( q->second, std::to_string(Idx) );

        return ts_info{};
    }

    std::vector<ts_info> find(const std::string & match, const queries_t & queries = queries_t{}) const {
        auto q = queries.find("value");
        FAST_REQUIRE_NE( q, queries.cend() );
        FAST_CHECK_EQ( q->first, std::string{"value"} );
        FAST_CHECK_EQ( q->second, std::to_string(Idx) );

        return std::vector<ts_info>{};
    };
};

struct multicontainer_test_dtss_dispatcher {

    using queries_t = std::map<std::string, std::string>;
    using container_wrapper_t = dtss::container_wrapper<
        multicontainer_test_dtss_container<0>, multicontainer_test_dtss_container<1> >;

    inline static const std::string container_query{ "container" };
    inline static const std::array<std::string, 2> remove_queries{{ "removed", container_query }};

    static void create_container(
        const std::string & container_name,
        const std::string & container_type,
        const std::string & root_path,
        shyft::dtss::server<multicontainer_test_dtss_dispatcher> & dtss_server
    ) {
        if ( container_type.empty() || container_type == "first" ) {
            dtss_server.container[std::string{"FIRST_"} + container_name] = container_wrapper_t{ std::make_unique<multicontainer_test_dtss_container<0>>(root_path) };
        } else if ( container_type == "second" ) {
            dtss_server.container[std::string{"SECOND_"} + container_name] = container_wrapper_t{ std::make_unique<multicontainer_test_dtss_container<1>>(root_path) };
        } else {
            throw std::runtime_error{ std::string{"Cannot construct unknown container type: "} + container_type };
        }
    }
     
    static container_wrapper_t & get_container(
        const std::string & container_name, const std::string & container_query,
        dtss::server<multicontainer_test_dtss_dispatcher> & dtss_server
    ) {
        decltype(dtss_server.container)::iterator f;
        if ( container_query.empty() || container_query == "first" ) {
            f = dtss_server.container.find(std::string{"FIRST_"} + container_name);
        } else if ( container_query == "second" ) {
            f = dtss_server.container.find(std::string{"SECOND_"} + container_name);
        } else {
            throw std::runtime_error{ std::string{"Cannot construct unknown container type: "} + container_query };
        }

        if( f == std::end(dtss_server.container) )
            throw runtime_error(std::string{"Failed to find shyft container: "} + container_name);

        return f->second;
    }
};

}

TEST_CASE("dtss_server_with_multiple_containers") {

    shyft::core::calendar utc;

    using shyft::time_axis::generic_dt;
    using shyft::time_series::dd::apoint_ts;
    using shyft::time_series::ts_point_fx;
    using ts_vector_t = shyft::time_series::dd::ats_vector;
    using test_server_t = shyft::dtss::server<multicontainer_test_dtss_dispatcher>;
    using queries_t = std::map<std::string, std::string>;

    const int port = 40336;

    test_server_t srv{};
    srv.set_listening_port(port);
    srv.add_container("container", "foo/bar/first", "first");
    srv.add_container("container", "foo/bar/second", "second");
    srv.start_async();

    shyft::dtss::client cli{ std::string{"localhost:"} + std::to_string(port) };

    /* NOTE: `remove` and `get_ts_info` are not called from the server...
    */

    SUBCASE("Save dispatched to the correct container") {
        core::utctime t = utc.time(2016, 1, 1);
        core::utctimespan dt = core::deltahours(1);
size_t n = 10;
generic_dt ta{ t, dt, n };

ts_vector_t vec{};
vec.emplace_back(
    shyft::dtss::shyft_url("container", "foo/bar/baz", queries_t{ {"value", "0"}, {"container", "first"} }),
    apoint_ts{ ta, 0 * 10.0, ts_point_fx::POINT_INSTANT_VALUE });
vec.emplace_back(
    shyft::dtss::shyft_url("container", "foo/bar/baz", queries_t{ {"value", "1"}, {"container", "second"} }),
    apoint_ts{ ta, 1 * 10.0, ts_point_fx::POINT_INSTANT_VALUE });

cli.store_ts(vec, false, false);
    }

    SUBCASE("Read dispatched to the correct container") {
        utcperiod period{ utc.time(2016, 1, 1), utc.time(2017, 1, 1) };

        ts_vector_t vec{};
        vec.emplace_back(shyft::dtss::shyft_url("container", "foo/bar/baz", queries_t{ {"value", "0"}, {"container", "first"} }));
        vec.emplace_back(shyft::dtss::shyft_url("container", "foo/bar/baz", queries_t{ {"value", "1"}, {"container", "second"} }));

        auto res = cli.evaluate(vec, period, false, false);
    }

    SUBCASE("Find dispatched to the correct container") {
        cli.find(shyft::dtss::shyft_url("container", "foo/bar/baz", queries_t{ {"value", "0"}, {"container", "first"} }));
        cli.find(shyft::dtss::shyft_url("container", "foo/bar/baz", queries_t{ {"value", "1"}, {"container", "second"} }));
    }
}

TEST_CASE("dtss_krls_io_routines") {
    using krls_io = shyft::dtss::krls_pred_db_io;

    shyft::core::calendar utc;

    fs::path datapath = fs::temp_directory_path() / fs::unique_path();
    fs::create_directories(datapath);
    const std::string ffp = (datapath / fs::path{ "krls_datafile.b" }).string();
    std::unique_ptr<std::FILE, decltype(&std::fclose)> fh{ std::fopen(ffp.c_str(), "w+b"), &std::fclose };

    const std::string source_url = "shyft://container/path/to/ts";
    const auto period = shyft::core::utcperiod{ utc.time(2017, 1, 1), utc.time(2018, 1, 1) };
    const auto krls_dt = utc.DAY;
    const auto point_fx = shyft::time_series::ts_point_fx::POINT_INSTANT_VALUE;
    const std::size_t krls_dict_size = 100000;
    const double krls_tolerance = 0.0001;
    const double rbfk_gamma = 0.001;

    // create the default kernel
    const auto predictor = krls_io::create_rbf_file(fh.get(), source_url, period, krls_dt, point_fx, krls_dict_size, krls_tolerance, rbfk_gamma);

    // attempt to use io functions to read the file back, all with default skip

    FAST_CHECK_UNARY(krls_io::can_read_file(fh.get()));

    auto read_krls_header = krls_io::read_header(fh.get());
    FAST_CHECK_EQ(read_krls_header.scaling, krls_dt);
    FAST_CHECK_EQ(read_krls_header.tolerance, krls_tolerance);
    FAST_CHECK_EQ(read_krls_header.point_fx, point_fx);
    FAST_CHECK_EQ(read_krls_header.t_start, period.start);
    FAST_CHECK_EQ(read_krls_header.t_end, period.end);

    auto read_source_url = krls_io::read_source_url(fh.get());
    FAST_CHECK_EQ(read_source_url, source_url);

    auto read_kernel_type = krls_io::read_predictor_kernel_type(fh.get());
    FAST_CHECK_EQ(read_kernel_type, shyft::dtss::krls_kernel_type_identifiers::radial_basis_kernel);

    auto read_rbf_header = krls_io::read_predictor_rbf_header(fh.get());
    FAST_CHECK_EQ(read_rbf_header.gamma, rbfk_gamma);

    auto read_predictor = krls_io::read_predictor_rbf_predictor(fh.get());
    FAST_CHECK_EQ(read_predictor.get_predictor_dt(), predictor.get_predictor_dt());
    FAST_CHECK_EQ(read_predictor.get_tolerance(), predictor.get_tolerance());
    FAST_CHECK_EQ(read_predictor.get_dictionary_size(), predictor.get_dictionary_size());
    FAST_CHECK_EQ(read_predictor.get_max_dictionary_size(), predictor.get_max_dictionary_size());
    FAST_CHECK_EQ(read_predictor.get_rbf_gamma(), predictor.get_rbf_gamma());
    FAST_CHECK_EQ(read_predictor.get_predicted_ts_point_policy(), predictor.get_predicted_ts_point_policy());
}

TEST_CASE("dtss_krls_db_register_update_read_rbf_series") {
    using calendar = shyft::core::calendar;
    using krls_db_t = shyft::dtss::krls_pred_db;
    using ts_vector_t = shyft::time_series::dd::ats_vector;
    using utcperiod = shyft::core::utcperiod;
    using utctime = shyft::core::utctime;

    calendar utc;

    fs::path krls_server_path = fs::temp_directory_path() / fs::unique_path();

    const std::string krls_name = "test-series";
    const std::string source_url = "test-series-url";
    const auto earliest_time = utc.time(2017, 1, 1);
    const auto middle_time = utc.time(2017, 3, 1);
    const auto latest_time = utc.time(2017, 5, 1);
    auto period = utcperiod{ earliest_time, middle_time };
    const auto krls_dt = 6*calendar::HOUR;
    const auto point_fx = shyft::time_series::ts_point_fx::POINT_INSTANT_VALUE;
    const std::size_t krls_dict_size = 1000000;
    const double krls_tolerance = 0.001;
    const double rbfk_gamma = 0.001;

    // server callback
    int data_dt = 2;
    int callback_call_count = 0;
    double data_offset = 0;
    std::function<ts_vector_t(const std::string &, utcperiod, bool, bool)> server_read_cb =
        [&callback_call_count, &period, &data_offset, point_fx, data_dt](const std::string &, utcperiod, bool, bool) -> ts_vector_t
    {
        callback_call_count += 1;

        // construct time axis
        std::size_t n = (period.end - period.start)/(data_dt*calendar::HOUR);
        shyft::time_axis::fixed_dt ta{ period.start, data_dt*calendar::HOUR, n };
        // generate values
        std::vector<double> values{};
        values.reserve(n);
        for ( std::size_t i = 0; i < n; ++i ) {
            values.emplace_back(data_offset + callback_call_count*i);
        }
        // setup return
        ts_vector_t result{};
        result.reserve(1);
        result.emplace_back(ta, values, point_fx);
        return result;
    };
    
    // setup the krls container
    krls_db_t krls_db{ krls_server_path.string(), server_read_cb };

    // register a new series
    krls_db.register_rbf_series(krls_name, source_url, period, krls_dt, point_fx, krls_dict_size, krls_tolerance, rbfk_gamma);
    // --------------------
    FAST_CHECK_EQ( callback_call_count, 1 );
    FAST_CHECK_UNARY( fs::exists(krls_server_path / krls_name) );

    // predict using the written data
    std::size_t n = (period.end - period.start)/calendar::HOUR;
    shyft::time_axis::generic_dt predict_ta{ period.start, calendar::HOUR, n };
    auto result_ts = krls_db.predict_time_series(krls_name, predict_ta);
    // --------------------
    FAST_CHECK_EQ( callback_call_count, 1 );
    FAST_REQUIRE_EQ( n, result_ts.size() );
    for ( std::size_t i = 10; i < n; ++i ) {  // skip 10 samples because of initial error
        FAST_CHECK_EQ( result_ts.value(i), doctest::Approx(callback_call_count*i/double(data_dt)).epsilon(0.1).scale(1) );  // epsilon is percent difference
    }
    data_offset = result_ts.value(n-1);

    // update the data with one more year
    period = utcperiod{ middle_time, latest_time };
    krls_db.update_rbf_series(krls_name, period);
    FAST_CHECK_EQ( callback_call_count, 2 );
    // --------------------
    n = (period.end - period.start)/calendar::HOUR;
    predict_ta = shyft::time_axis::generic_dt{ period.start, calendar::HOUR, n };
    result_ts = krls_db.predict_time_series(krls_name, predict_ta);
    // --------------------
    FAST_REQUIRE_EQ( n, result_ts.size() );
    for ( std::size_t i = 10; i < n; ++i ) {  // skip 10 samples because of initial error
        FAST_CHECK_EQ( result_ts.value(i), doctest::Approx(data_offset + callback_call_count*i/double(data_dt)).epsilon(0.1).scale(1) );  // epsilon is percent difference
    }
}

TEST_CASE("dtss_krls_db_register_update_read_rbf_series_using_save_read") {
    using calendar = shyft::core::calendar;
    using gta_t = shyft::time_axis::generic_dt;
    using gts_t = shyft::time_series::point_ts<gta_t>;
    using krls_db_t = shyft::dtss::krls_pred_db;
    using ts_vector_t = shyft::time_series::dd::ats_vector;
    using utcperiod = shyft::core::utcperiod;
    using utctime = shyft::core::utctime;

    calendar utc;

    fs::path krls_server_path = fs::temp_directory_path() / fs::unique_path();

    const std::string krls_name = "test-series";
    const std::string source_url = "test-series-url";
    const auto earliest_time = utc.time(2017, 1, 1);
    const auto middle_time = utc.time(2017, 3, 1);
    const auto latest_time = utc.time(2017, 5, 1);
    auto period = utcperiod{ earliest_time, middle_time };
    const auto krls_dt = 6*calendar::HOUR;
    const auto point_fx = shyft::time_series::ts_point_fx::POINT_INSTANT_VALUE;
    const std::size_t krls_dict_size = 1000000;
    const double krls_tolerance = 0.001;
    const double rbfk_gamma = 0.001;

    // server callback
    int data_dt = 2;
    int callback_call_count = 0;
    double data_offset = 0;
    std::function<ts_vector_t(const std::string &, utcperiod, bool, bool)> server_read_cb =
        [&callback_call_count, &period, &data_offset, point_fx, data_dt](const std::string &, utcperiod, bool, bool) -> ts_vector_t
    {
        callback_call_count += 1;

        // construct time axis
        std::size_t n = (period.end - period.start)/(data_dt*calendar::HOUR);
        gta_t ta{ period.start, data_dt*calendar::HOUR, n };
        // generate values
        std::vector<double> values{};
        values.reserve(n);
        for ( std::size_t i = 0; i < n; ++i ) {
            values.emplace_back(data_offset + callback_call_count*i);
        }
        // setup return
        ts_vector_t result{};
        result.reserve(1);
        result.emplace_back(ta, values, point_fx);
        return result;
    };

    // setup the krls container
    krls_db_t krls_db{ krls_server_path.string(), server_read_cb };

    // register a new series
    std::size_t n = (period.end - period.start)/calendar::HOUR;
    krls_db.save(krls_name,
        gts_t{ gta_t{ period.start, calendar::HOUR, n }, 0., point_fx }, false,
        krls_db_t::queries_t{
            {"source_url", source_url},
            {"dt_scaling", std::to_string(krls_dt)},
            {"point_fx", "instant"},
            {"krls_dict_size", std::to_string(krls_dict_size)},
            {"tolerance", std::to_string(krls_tolerance)},
            {"gamma", std::to_string(rbfk_gamma)}
        });
    // --------------------
    FAST_CHECK_EQ( callback_call_count, 1 );
    FAST_CHECK_UNARY( fs::exists(krls_server_path / krls_name) );

    // predict using the written data
    auto result_ts = krls_db.read(krls_name, period, krls_db_t::queries_t{
            {"dt", std::to_string(calendar::HOUR)},
        });
    // --------------------
    FAST_CHECK_EQ( callback_call_count, 1 );
    FAST_REQUIRE_EQ( n, result_ts.size() );
    for ( std::size_t i = 10; i < n; ++i ) {  // skip 10 samples because of initial error
        FAST_CHECK_EQ( result_ts.value(i), doctest::Approx(callback_call_count*i/double(data_dt)).epsilon(0.1).scale(1) );  // epsilon is percent difference
    }
    data_offset = result_ts.value(n-1);

    // update the data with one more year
    period = utcperiod{ middle_time, latest_time };
    n = (period.end - period.start)/calendar::HOUR;
    krls_db.save(krls_name, gts_t{ gta_t{ period.start, calendar::HOUR, n }, 0., point_fx },
        false, krls_db_t::queries_t{});
    FAST_CHECK_EQ( callback_call_count, 2 );
    // --------------------
    n = (period.end - period.start)/calendar::HOUR;
    result_ts = krls_db.read(krls_name, period, krls_db_t::queries_t{
            {"dt", std::to_string(calendar::HOUR)},
        });
    // --------------------
    FAST_REQUIRE_EQ( n, result_ts.size() );
    for ( std::size_t i = 10; i < n; ++i ) {  // skip 10 samples because of initial error
        FAST_CHECK_EQ( result_ts.value(i), doctest::Approx(data_offset + callback_call_count*i/double(data_dt)).epsilon(0.1).scale(1) );  // epsilon is percent difference
    }
}

}
