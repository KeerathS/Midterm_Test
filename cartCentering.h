#ifndef cartCentering_h
#define cartCentering_h

#include <math.h>
#include <stdlib.h>
#include <unistd.h>  // usleep

#include <iomanip>  // setw
#include <iostream>
#include <random>
#include <vector>

#define NEARZERO 10e-12
inline bool isEqual(double x, double y) { return fabs(x - y) < NEARZERO; }

/******************************************************************************/
class cartCentering {
 protected:
  // parameters for simulation
  const double MASSCART = 2.0;
  const double FORCE_MAG = 1.0;
  const double TAU = 0.02;  // seconds between state updates

  // these may depend on each other and dt
  const double MAX_X = 1.5;
  const double MAX_V = 6;

  const double MIN_VAR_INI = -0.75;
  const double MAX_VAR_INI = 0.75;
  const double NEAR_ORIGIN = 0.01;

  // system state
  const int STATE_SIZE = 2;
  const int X = 0;
  const int V = 1;
  std::vector<double> state;

  int step;      // current simulation step
  int max_step;  // max simulation steps
  std::uniform_real_distribution<> disReset;
  bool draw_track = false;

 public:
  /************************************************************************/
  cartCentering() {
    disReset = std::uniform_real_distribution<>(MIN_VAR_INI, MAX_VAR_INI);
    max_step = 500;
    state.resize(STATE_SIZE);
  }

  /************************************************************************/
  ~cartCentering() {}

  /************************************************************************/
  void reset(std::mt19937& rng) {
    step = 0;
    do {
      state[X] = disReset(rng);
      state[V] = disReset(rng);
    } while (terminal());
  }

  /************************************************************************/
  bool terminal() {
    if (step >= max_step)
      return true;
    else if (abs(state[X]) <= NEAR_ORIGIN && abs(state[V]) <= NEAR_ORIGIN)
      return true;
    else if (abs(state[X]) > MAX_X)
      return true;
    return false;
  }

  /************************************************************************/
  double update(const int& action, bool animate = false) {
    double force = action < 0 ? -FORCE_MAG : FORCE_MAG;
    double acc_t = force / MASSCART;

    state[X] += TAU * state[V];
    state[V] += TAU * acc_t;
    state[V] = bound(state[V], -MAX_V, MAX_V);
    step++;
    if (animate) draw(action);
    if (terminal()) {
      double x = (abs(state[X]) / MAX_X) * 1.0;
      double v = (abs(state[V]) / MAX_V) * 0.5;
      double s = ((double)step / max_step) * 0.25;
      return -(x + v + s);
    } else
      return 0;
  }

  /************************************************************************/
  double bound(double x, double m, double M) {
    return std::min(std::max(x, m), M);
  }
  double getCartXPos() { return state[X]; }
  double getCartXVel() { return state[V]; }
  void setDraw(bool d) { draw_track = d; }

  /************************************************************************/
  void draw(const int& action) {
    clearScreen();
    bool solved =
        (abs(state[X]) <= NEAR_ORIGIN && abs(state[V]) <= NEAR_ORIGIN);

    std::cout << "Step: " << step << std::endl;
    std::cout << "X " << std::setprecision(3) << state[X] << std::endl;
    std::cout << "V " << std::setprecision(3) << state[V] << std::endl;
    std::cout << "Action: " << (action < 0 ? "<--" : "-->") << std::endl;
    if (terminal())
      std::cout << "Solved: " << (solved ? "YES!" : "NO") << std::endl;
    else
      std::cout << "Solved:" << std::endl;

    const int track_length = 121;
    // map state[X] from range (-1.5, 1.5) to (0, 120)
    int x = int((state[X] + MAX_X) * 40);

    // draw cart
    std::string s = std::string(track_length, ' ');
    s.replace(x, 1, "*");
    std::cout << s << std::endl;

    // draw track
    s = std::string(track_length, '_');
    s.replace(60, 1, "|");  // center position
    std::cout << s << std::endl;
    std::cout << std::flush;

    if (terminal())
      sleep(3);
    else
      usleep(50000);
   clearScreen();
  }
  
  void clearScreen() {
    printf("\033[2J");
    printf("\033[%d;%dH", 0, 0);
  }
};
#endif
