// x2p_cmd_relay: the joint command writer for the x2-perception deploy node, in C++.
//
// Why (docs/FINDINGS.md X9, X10): the Python deploy node's own 500 Hz writer reached the bus at 208 Hz with gaps up to 21 ms,
// and the real legs then delivered ~0.65 of the commanded stiffness, against 0.96-0.98 under the vendor MC's steady ~490 Hz
// stream. This node takes that job away from Python, as SONIC's C++ deploy does with its dedicated 500 Hz writer thread: the
// Python node publishes each new command once per 50 Hz control step on <in_prefix>/<group>/command, and this node republishes
// the latest one of every group to <out_prefix>/<group>/command at a steady rate with a fresh stamp.
//
// Safety:
//  * Nothing is published before the first command arrives (MC may own the bus; a dry-run / --no-publish node never feeds us).
//  * If no command arrives for stale_s (default 0.2 s: the Python node died or hung), every joint goes limp: stiffness 0,
//    damping as last commanded, target = last target. A stale stiff target would otherwise be held forever.
//  * Every 5 s it logs its input and output rates and the largest output gap.
#include <aimdk_msgs/msg/joint_command_array.hpp>
#include <rclcpp/rclcpp.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>

using JointCommandArray = aimdk_msgs::msg::JointCommandArray;
using Clock = std::chrono::steady_clock;

namespace {
const std::array<const char*, 4> kGroups = {"leg", "waist", "arm", "head"};
}

class CmdRelay : public rclcpp::Node {
 public:
  CmdRelay() : Node("x2p_cmd_relay") {
    rate_hz_ = declare_parameter<double>("rate_hz", 500.0);
    stale_s_ = declare_parameter<double>("stale_s", 0.2);
    const auto in_prefix = declare_parameter<std::string>("in_prefix", "/x2p/relay");
    const auto out_prefix = declare_parameter<std::string>("out_prefix", "/aima/hal/joint");
    const auto qos = rclcpp::SensorDataQoS();
    for (size_t g = 0; g < kGroups.size(); ++g) {
      pubs_[g] = create_publisher<JointCommandArray>(out_prefix + "/" + kGroups[g] + "/command", qos);
      subs_[g] = create_subscription<JointCommandArray>(
          in_prefix + "/" + kGroups[g] + "/command", qos, [this, g](JointCommandArray::SharedPtr m) {
            std::lock_guard<std::mutex> lk(mu_);
            latest_[g] = *m;
            have_[g] = true;
            last_in_ = Clock::now();
            n_in_++;
          });
    }
    RCLCPP_INFO(get_logger(), "%s/<group>/command -> %s/<group>/command at %.0f Hz; limp after %.2f s without input",
                in_prefix.c_str(), out_prefix.c_str(), rate_hz_, stale_s_);
    writer_ = std::thread([this] { Run(); });
  }

  ~CmdRelay() override {
    stop_ = true;
    if (writer_.joinable()) writer_.join();
  }

 private:
  void Run() {
    const auto period = std::chrono::nanoseconds(static_cast<int64_t>(1e9 / rate_hz_));
    auto next = Clock::now();
    auto t_log = next, t_prev = next;
    double max_gap_ms = 0.0;
    long n_out = 0, n_in_prev = 0;
    bool was_stale = false;
    while (!stop_ && rclcpp::ok()) {
      next += period;
      std::this_thread::sleep_until(next);
      const auto now = Clock::now();
      std::array<JointCommandArray, 4> out;
      std::array<bool, 4> have{};
      bool stale = false;
      {
        std::lock_guard<std::mutex> lk(mu_);
        have = have_;
        out = latest_;
        stale = std::chrono::duration<double>(now - last_in_).count() > stale_s_;
      }
      bool any = false;
      for (bool h : have) any = any || h;
      if (!any) {                                   // never commanded: stay off the bus
        t_prev = now;
        continue;
      }
      if (stale != was_stale) {
        if (stale) RCLCPP_WARN(get_logger(), "no command for %.2f s: limp (stiffness 0)", stale_s_);
        else RCLCPP_INFO(get_logger(), "commands again: relaying");
        was_stale = stale;
      }
      const auto stamp = get_clock()->now();
      for (size_t g = 0; g < kGroups.size(); ++g) {
        if (!have[g]) continue;
        auto& m = out[g];
        m.header.stamp = stamp;
        if (stale)
          for (auto& j : m.joints) j.stiffness = 0.0;
        pubs_[g]->publish(m);
      }
      n_out++;
      const double gap = std::chrono::duration<double, std::milli>(now - t_prev).count();
      if (gap > max_gap_ms) max_gap_ms = gap;
      t_prev = now;
      if (now - t_log > std::chrono::seconds(5)) {
        const double dt = std::chrono::duration<double>(now - t_log).count();
        const long n_in = n_in_.load();
        RCLCPP_INFO(get_logger(), "out %.0f Hz (max gap %.1f ms), in %.0f msgs/s%s", n_out / dt, max_gap_ms,
                    (n_in - n_in_prev) / dt, stale ? ", LIMP (no input)" : "");
        n_out = 0;
        n_in_prev = n_in;
        max_gap_ms = 0.0;
        t_log = now;
      }
      if (Clock::now() - next > std::chrono::milliseconds(50)) next = Clock::now();   // never burst to catch up
    }
  }

  double rate_hz_{500.0}, stale_s_{0.2};
  std::array<rclcpp::Publisher<JointCommandArray>::SharedPtr, 4> pubs_;
  std::array<rclcpp::Subscription<JointCommandArray>::SharedPtr, 4> subs_;
  std::mutex mu_;
  std::array<JointCommandArray, 4> latest_;
  std::array<bool, 4> have_{};
  Clock::time_point last_in_{Clock::now()};
  std::atomic<long> n_in_{0};
  std::atomic<bool> stop_{false};
  std::thread writer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CmdRelay>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
