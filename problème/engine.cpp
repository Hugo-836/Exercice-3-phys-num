// =============================================================================
//  engine.cpp  —  Simulateur gravitationnel N corps (N ≤ 3), 2D cartésien
//  Exercice 3, Physique Numérique EPFL 2026
//
//  Vecteur d'état (taille 4N) :
//    y = (x1, y1, …, xN, yN,  vx1, vy1, …, vxN, vyN)
//
//  Helpers d'index :
//    ix(i) = 2i          iy(i) = 2i+1
//    ivx(i) = 2N+2i      ivy(i) = 2N+2i+1
//
//  Colonnes de sortie (séparées par des espaces) :
//    t   x1 y1 … xN yN   vx1 vy1 … vxN vyN   E  px py  dt
//    [+ a_drag  P_drag si traînée activée]
//
//  Paramètres du fichier de configuration (% commence un commentaire) :
//    N, G, tf
//    mass_i, radius_i, x_i, y_i, vx_i, vy_i   (i = 1…N, 1-indexé)
//    adaptive  (bool)
//    nsteps    (entier, mode dt fixe)
//    dt0, epsilon, dtMin, dtMax  (mode adaptatif)
//    drag, rho0, lambda_atm, S, Cx, drag_body, drag_center
//    auto_init, r0_artemis, v0_artemis, h_perigee, frame_correction
//    sampling, output
// =============================================================================

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <valarray>
#include <vector>

#include "../common/ConfigFile.h"

// =============================================================================
class Engine
{
public:
    explicit Engine(const ConfigFile& cfg);
    void run();

private:
    // ── Helpers d'index ──────────────────────────────────────────────────────
    std::size_t ix (std::size_t i) const { return 2 * i; }
    std::size_t iy (std::size_t i) const { return 2 * i + 1; }
    std::size_t ivx(std::size_t i) const { return 2 * N_ + 2 * i; }
    std::size_t ivy(std::size_t i) const { return 2 * N_ + 2 * i + 1; }

    // ── Physique ─────────────────────────────────────────────────────────────
    std::valarray<double> computeRhs(const std::valarray<double>& y) const;
    std::valarray<double> rk4Step   (double h, const std::valarray<double>& y) const;

    double mechEnergy    (const std::valarray<double>& y) const;
    void   totalMomentum (const std::valarray<double>& y, double& px, double& py) const;
    double dragAccelMag  (const std::valarray<double>& y) const;
    double dragPower     (const std::valarray<double>& y) const;
    bool   checkCollision(const std::valarray<double>& y) const;

    // ── Intégration temporelle ────────────────────────────────────────────────
    bool adaptiveStep(double& dt);   // retourne true si le pas est accepté
    void fixedStep();

    // ── Sortie ───────────────────────────────────────────────────────────────
    void writeState(bool forceWrite);

    // ── Paramètres ───────────────────────────────────────────────────────────
    std::size_t         N_      = 0;
    std::vector<double> mass_;       // [N_] kg
    std::vector<double> radius_;     // [N_] m  (rayon physique, détection collision)

    double G_  = 6.674e-11;
    double tf_ = 0.0;

    // Mode dt fixe
    int    nsteps_  = 0;
    double dtFixed_ = 0.0;

    // Mode dt adaptatif
    bool   adaptive_ = false;
    double dt0_      = 10.0;
    double epsilon_  = 1e-8;   // tolérance mixte abs+rel par pas
    double dtMin_    = 1e-2;
    double dtMax_    = 1e6;

    // Traînée (optionnelle)
    bool        drag_        = false;
    double      rho0_        = 0.0;
    double      lambdaAtm_   = 0.0;
    double      S_           = 0.0;
    double      Cx_          = 0.0;
    std::size_t dragBody_    = 1;    // 0-indexé
    std::size_t dragCenter_  = 0;

    // Runtime
    double       t_                   = 0.0;
    double       dt_                  = 0.0;
    unsigned int sampling_            = 1;
    unsigned int stepsSinceLastWrite_ = 0;

    std::valarray<double> state_;
    std::ofstream         outputFile_;
};

// =============================================================================
//  Constructeur
// =============================================================================

Engine::Engine(const ConfigFile& cfg)
{
    G_  = cfg.get<double>("G",  6.674e-11);
    tf_ = cfg.get<double>("tf");

    N_ = static_cast<std::size_t>(cfg.get<int>("N", 2));
    if (N_ < 1 || N_ > 3)
        throw std::runtime_error("N doit etre 1, 2 ou 3");

    mass_  .resize(N_);
    radius_.resize(N_);
    state_.resize(4 * N_, 0.0);

    for (std::size_t i = 0; i < N_; ++i) {
        const std::string s = std::to_string(i + 1);
        mass_  [i] = cfg.get<double>("mass_"   + s);
        radius_[i] = cfg.get<double>("radius_" + s, 1.0);
        state_[ix (i)] = cfg.get<double>("x_"  + s, 0.0);
        state_[iy (i)] = cfg.get<double>("y_"  + s, 0.0);
        state_[ivx(i)] = cfg.get<double>("vx_" + s, 0.0);
        state_[ivy(i)] = cfg.get<double>("vy_" + s, 0.0);
    }

    adaptive_ = cfg.get<bool>("adaptive", false);
    if (adaptive_) {
        dt0_     = cfg.get<double>("dt0",     10.0);
        epsilon_ = cfg.get<double>("epsilon", 1e-8);
        dtMin_   = cfg.get<double>("dtMin",   1e-2);
        dtMax_   = cfg.get<double>("dtMax",   1e6);
        dt_      = dt0_;
    } else {
        nsteps_  = cfg.get<int>("nsteps", 10000);
        if (nsteps_ <= 0)
            throw std::runtime_error("nsteps doit etre > 0");
        dtFixed_ = tf_ / static_cast<double>(nsteps_);
        dt_      = dtFixed_;
    }

    sampling_ = cfg.get<unsigned int>("sampling", 1U);

    drag_ = cfg.get<bool>("drag", false);
    if (drag_) {
        rho0_      = cfg.get<double>("rho0");
        lambdaAtm_ = cfg.get<double>("lambda_atm");
        S_         = cfg.get<double>("S");
        Cx_        = cfg.get<double>("Cx");
        dragBody_  = static_cast<std::size_t>(cfg.get<int>("drag_body",   2)) - 1;
        dragCenter_= static_cast<std::size_t>(cfg.get<int>("drag_center", 1)) - 1;
        if (dragBody_ >= N_ || dragCenter_ >= N_)
            throw std::runtime_error("drag_body ou drag_center hors intervalle");
    }

    // ── Auto-init : calcule analytiquement (vr, vt) pour le périgée désiré ──
    // Artemis = dernier corps (indice N_-1), Terre = corps 0.
    if (cfg.get<bool>("auto_init", false)) {
        const double r0  = cfg.get<double>("r0_artemis");
        const double v0  = cfg.get<double>("v0_artemis");
        const double h   = cfg.get<double>("h_perigee");
        const double RT  = radius_[0];
        const double mu  = G_ * mass_[0];
        const double rp  = RT + h;

        const double Espec = 0.5 * v0 * v0 - mu / r0;
        if (Espec >= 0.0)
            throw std::runtime_error("auto_init: orbite non elliptique");

        const double a   = -mu / (2.0 * Espec);
        const double ecc = 1.0 - rp / a;
        if (ecc < 0.0 || ecc >= 1.0)
            throw std::runtime_error("auto_init: excentricite invalide");

        const double L    = std::sqrt(mu * a * (1.0 - ecc * ecc));
        const double vt   = L / r0;
        const double vr2  = v0 * v0 - vt * vt;
        if (vr2 < 0.0)
            throw std::runtime_error("auto_init: vitesse tangentielle > v0");

        const double vr   = -std::sqrt(vr2);
        const double vmax =  std::sqrt(mu * (2.0 / rp - 1.0 / a));

        const std::size_t iA = N_ - 1;
        state_[ix (iA)] = r0;
        state_[iy (iA)] = 0.0;
        state_[ivx(iA)] = vr;
        state_[ivy(iA)] = vt;

        // Correction barycentrique (section 3.5) : ajouter la vitesse de la Terre
        if (cfg.get<bool>("frame_correction", false)) {
            state_[ix (iA)] += state_[ix (0)];
            state_[iy (iA)] += state_[iy (0)];
            state_[ivx(iA)] += state_[ivx(0)];
            state_[ivy(iA)] += state_[ivy(0)];
        }

        std::cout << "=== Conditions initiales (auto_init) ===\n"
                  << "  a      = " << a    << " m\n"
                  << "  e      = " << ecc  << "\n"
                  << "  rp     = " << rp   << " m  (h=" << (rp-RT) << " m)\n"
                  << "  vr(0)  = " << vr   << " m/s\n"
                  << "  vt(0)  = " << vt   << " m/s\n"
                  << "  vmax   = " << vmax << " m/s\n";
    }

    const std::string outPath = cfg.get<std::string>("output", "output.out");
    outputFile_.open(outPath);
    if (!outputFile_.is_open())
        throw std::runtime_error("Impossible d'ouvrir: " + outPath);
    outputFile_ << std::scientific << std::setprecision(10);
}

// =============================================================================
//  dy/dt = f(y)
// =============================================================================

std::valarray<double> Engine::computeRhs(const std::valarray<double>& y) const
{
    std::valarray<double> dydt(0.0, 4 * N_);

    // d(positions)/dt = vitesses
    for (std::size_t i = 0; i < N_; ++i) {
        dydt[ix (i)] = y[ivx(i)];
        dydt[iy (i)] = y[ivy(i)];
    }

    // d(vitesses)/dt = forces gravitationnelles (paires, 3e loi de Newton)
    for (std::size_t i = 0; i < N_; ++i) {
        for (std::size_t j = i + 1; j < N_; ++j) {
            const double dx  = y[ix(j)] - y[ix(i)];
            const double dy_ = y[iy(j)] - y[iy(i)];
            const double r2  = dx * dx + dy_ * dy_;
            const double r   = std::sqrt(r2);
            const double r3  = r2 * r;

            const double Gij = G_ / r3;
            dydt[ivx(i)] += Gij * mass_[j] * dx;
            dydt[ivy(i)] += Gij * mass_[j] * dy_;
            dydt[ivx(j)] -= Gij * mass_[i] * dx;
            dydt[ivy(j)] -= Gij * mass_[i] * dy_;
        }
    }

    // Traînée aérodynamique
    if (drag_) {
        const std::size_t db  = dragBody_;
        const std::size_t dcb = dragCenter_;

        const double dxr = y[ix(db)]  - y[ix(dcb)];
        const double dyr = y[iy(db)]  - y[iy(dcb)];
        const double r   = std::sqrt(dxr * dxr + dyr * dyr);

        const double dvx = y[ivx(db)] - y[ivx(dcb)];
        const double dvy = y[ivy(db)] - y[ivy(dcb)];
        const double spd = std::sqrt(dvx * dvx + dvy * dvy);

        const double rho  = rho0_ * std::exp(-(r - radius_[dcb]) / lambdaAtm_);
        const double fact = -0.5 * rho * S_ * Cx_ * spd / mass_[db];

        dydt[ivx(db)] += fact * dvx;
        dydt[ivy(db)] += fact * dvy;
    }

    return dydt;
}

// =============================================================================
//  Un pas RK4
// =============================================================================

std::valarray<double> Engine::rk4Step(double h, const std::valarray<double>& y) const
{
    const std::valarray<double> k1 = computeRhs(y);
    const std::valarray<double> k2 = computeRhs(y + (0.5 * h) * k1);
    const std::valarray<double> k3 = computeRhs(y + (0.5 * h) * k2);
    const std::valarray<double> k4 = computeRhs(y + h * k3);
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

// =============================================================================
//  Pas fixe
// =============================================================================

void Engine::fixedStep()
{
    state_ = rk4Step(dt_, state_);
    t_ += dt_;
}

// =============================================================================
//  Pas adaptatif — doublement de pas (extrapolation de Richardson)
//
//  Un grand pas h  → y_coarse   (erreur ~ C h^5)
//  Deux demi-pas   → y_fine     (erreur ~ C h^5 / 16)
//  Estimation :  δ_i = |y_fine[i] - y_coarse[i]| / 15
//  Erreur normalisée : err = max_i  δ_i / (ε*(1+|y_fine[i]|))
//  Accepté si err ≤ 1 ;  nouveau pas ← h * 0.9 * (1/err)^(1/5)
// =============================================================================

bool Engine::adaptiveStep(double& dt)
{
    const double actualDt = std::min(dt, tf_ - t_);

    const std::valarray<double> y_coarse = rk4Step(actualDt,       state_);
    const std::valarray<double> y_mid    = rk4Step(0.5*actualDt,   state_);
    const std::valarray<double> y_fine   = rk4Step(0.5*actualDt,   y_mid);

    double err = 0.0;
    for (std::size_t i = 0; i < state_.size(); ++i) {
        const double scale = epsilon_ * (1.0 + std::abs(y_fine[i]));
        const double di    = std::abs(y_fine[i] - y_coarse[i]) / (15.0 * scale);
        if (di > err) err = di;
    }
    if (err < 1e-30) err = 1e-30;

    const double factor = std::min(5.0, std::max(0.1, 0.9 * std::pow(1.0/err, 0.2)));
    const double newDt  = std::min(std::max(actualDt * factor, dtMin_), dtMax_);

    if (err <= 1.0 || actualDt <= dtMin_) {
        state_ = y_fine;
        t_    += actualDt;
        dt     = newDt;
        return true;
    } else {
        dt = newDt;
        return false;
    }
}

// =============================================================================
//  Diagnostics
// =============================================================================

double Engine::mechEnergy(const std::valarray<double>& y) const
{
    double E = 0.0;
    for (std::size_t i = 0; i < N_; ++i) {
        const double vx = y[ivx(i)], vy = y[ivy(i)];
        E += 0.5 * mass_[i] * (vx*vx + vy*vy);
    }
    for (std::size_t i = 0; i < N_; ++i) {
        for (std::size_t j = i+1; j < N_; ++j) {
            const double dx  = y[ix(j)] - y[ix(i)];
            const double dy_ = y[iy(j)] - y[iy(i)];
            E -= G_ * mass_[i] * mass_[j] / std::sqrt(dx*dx + dy_*dy_);
        }
    }
    return E;
}

void Engine::totalMomentum(const std::valarray<double>& y, double& px, double& py) const
{
    px = py = 0.0;
    for (std::size_t i = 0; i < N_; ++i) {
        px += mass_[i] * y[ivx(i)];
        py += mass_[i] * y[ivy(i)];
    }
}

double Engine::dragAccelMag(const std::valarray<double>& y) const
{
    if (!drag_) return 0.0;
    const std::size_t db = dragBody_, dcb = dragCenter_;
    const double dxr = y[ix(db)] -y[ix(dcb)], dyr = y[iy(db)] -y[iy(dcb)];
    const double r   = std::sqrt(dxr*dxr + dyr*dyr);
    const double dvx = y[ivx(db)]-y[ivx(dcb)], dvy = y[ivy(db)]-y[ivy(dcb)];
    const double spd = std::sqrt(dvx*dvx + dvy*dvy);
    return 0.5 * rho0_ * std::exp(-(r-radius_[dcb])/lambdaAtm_) * S_ * Cx_ * spd*spd / mass_[db];
}

double Engine::dragPower(const std::valarray<double>& y) const
{
    if (!drag_) return 0.0;
    const std::size_t db = dragBody_, dcb = dragCenter_;
    const double dxr = y[ix(db)] -y[ix(dcb)], dyr = y[iy(db)] -y[iy(dcb)];
    const double r   = std::sqrt(dxr*dxr + dyr*dyr);
    const double dvx = y[ivx(db)]-y[ivx(dcb)], dvy = y[ivy(db)]-y[ivy(dcb)];
    const double spd = std::sqrt(dvx*dvx + dvy*dvy);
    return -0.5 * rho0_ * std::exp(-(r-radius_[dcb])/lambdaAtm_) * S_ * Cx_ * spd*spd*spd;
}

bool Engine::checkCollision(const std::valarray<double>& y) const
{
    for (std::size_t i = 0; i < N_; ++i)
        for (std::size_t j = i+1; j < N_; ++j) {
            const double dx  = y[ix(j)]-y[ix(i)], dy_ = y[iy(j)]-y[iy(i)];
            if (std::sqrt(dx*dx+dy_*dy_) < radius_[i]+radius_[j]) return true;
        }
    return false;
}

// =============================================================================
//  Écriture
// =============================================================================

void Engine::writeState(bool forceWrite)
{
    if (!forceWrite) {
        if (stepsSinceLastWrite_ < sampling_) { ++stepsSinceLastWrite_; return; }
    }
    outputFile_ << t_;
    for (std::size_t i = 0; i < N_; ++i)
        outputFile_ << ' ' << state_[ix(i)] << ' ' << state_[iy(i)];
    for (std::size_t i = 0; i < N_; ++i)
        outputFile_ << ' ' << state_[ivx(i)] << ' ' << state_[ivy(i)];
    double px, py;
    totalMomentum(state_, px, py);
    outputFile_ << ' ' << mechEnergy(state_)
                << ' ' << px << ' ' << py
                << ' ' << dt_;
    if (drag_)
        outputFile_ << ' ' << dragAccelMag(state_) << ' ' << dragPower(state_);
    outputFile_ << '\n';
    stepsSinceLastWrite_ = 1;
}

// =============================================================================
//  Boucle principale
// =============================================================================

void Engine::run()
{
    t_  = 0.0;
    dt_ = adaptive_ ? dt0_ : dtFixed_;
    stepsSinceLastWrite_ = sampling_;
    writeState(false);

    if (adaptive_) {
        while (t_ < tf_ - 1e-12 * tf_) {
            while (!adaptiveStep(dt_)) {}
            writeState(false);
            if (checkCollision(state_)) {
                std::cout << "Collision a t = " << t_ << " s\n";
                break;
            }
        }
    } else {
        for (int step = 0; step < nsteps_; ++step) {
            fixedStep();
            writeState(false);
            if (checkCollision(state_)) {
                std::cout << "Collision a t = " << t_ << " s\n";
                break;
            }
        }
    }
    writeState(true);
    std::cout << "Simulation terminee. t = " << t_ << " s\n";
}

// =============================================================================
//  main
// =============================================================================

int main(int argc, char* argv[])
{
    try {
        std::string inputPath = "configuration.in.example";
        if (argc > 1) inputPath = argv[1];
        ConfigFile cfg(inputPath);
        for (int i = 2; i < argc; ++i) cfg.process(argv[i]);
        Engine engine(cfg);
        engine.run();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Erreur: " << e.what() << '\n';
        return 1;
    }
}
