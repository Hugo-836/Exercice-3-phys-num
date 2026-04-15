#include <iostream>       // basic input output streams
#include <fstream>        // input output file stream class
#include <cmath>          // librerie mathematique de base
#include <iomanip>        // input output manipulators
#include <valarray>       // for std::valarray
#include "../common/ConfigFile.h" // Il contient les methodes pour lire inputs et ecrire outputs 
#include <numeric>

using namespace std; // ouvrir un namespace avec la librerie c++ de base

/* La class Engine est le moteur principale de ce code. Il contient 
   les methodes de base pour lire / initialiser les inputs, 
   preparer les outputs et calculer les donnees necessaires
*/
class Engine
{
private:
    // Existing private members of Engine...
  const double pi=3.1415926535897932384626433832795028841971e0;

  // definition des variables

  double G, mA, d, v0, h, RT, mT, r0;     // accélération gravitationnelle, masse, longueur, fréquence angulaire, rayon, coefficient de frottement

  valarray<double> vect;

  double t; // Temps courant pas de temps
  double tf;          // Temps final
  double dt;      // Intervalle de temps

  unsigned int sampling;  // Nombre de pas de temps entre chaque ecriture des diagnostics
  int nsteps_per;
  unsigned int last;       // Nombre de pas de temps depuis la derniere ecriture des diagnostics
  ofstream *outputFile;    // Pointeur vers le fichier de sortie

  /* Calculer et ecrire les diagnostics dans un fichier
     inputs:
     write: (bool) ecriture de tous les sampling si faux
  */  
  void printOut(bool write)
  {

    // Ecriture tous les [sampling] pas de temps, sauf si write est vrai
    if((!write && last>=sampling) || (write && last!=1))
    {
      *outputFile << t << " " << vect[0] << " " << vect[1] << " " << vect[2] << " " << vect[3] << endl;
      last = 1;
    }
    else
    {
      last++;
    }
  }

  // TODO écrire la fonction pour l'acceleration (theta_doubledot)
  valarray<double> compute_f(valarray<double> vect)
  {
      valarray<double> f(0.0,4);
      double r = vect[0]*vect[0] + vect[1]*vect[1];
      f[0] = vect[2];
      f[1] = vect[3];
      f[2] = - G*mT*vect[0]/pow(r,1.5);
      f[3] = - G*mT*vect[1]/pow(r,1.5);
      return f;
  }

  void step()
  {
    valarray <double> k1 = compute_f(vect);
    valarray <double> k2 = compute_f(vect+0.5*k1);
    valarray <double> k3 = compute_f(vect+0.5*k2);   
    valarray <double> k4 = compute_f(vect+k3);

    vect = vect + dt/6.0 * (k1 + 2*k2 + 2*k3 + k4);
    t = t + dt;
  }


public:
    // Modified constructor
    Engine(ConfigFile configFile)
    {
      // Stockage des parametres de simulation dans les attributs de la classe
      tf     = configFile.get<double>("tf",tf);	        // t final (overwritten if N_excit >0)
      G     = configFile.get<double>("G", G);         // lire l'acceleration de gravite
      mA     = configFile.get<double>("mA", mA);         // lire la masse
      d     = configFile.get<double>("d", d);         // lire la longueur
      r0  = configFile.get<double>("r0", r0);  
      v0 = configFile.get<double>("v0", v0); // lire la frequence angulaire
      h     = configFile.get<double>("h", h);         // lire le rayon
      RT = configFile.get<double>("RT", RT); // lire le coefficient de frottement
      mT    = configFile.get<double>("mT", mT);    // lire la condition initiale en theta
      vect = valarray<double>{r0,0,0,v0};
      nsteps_per= configFile.get<int>("nsteps");        // number of time step per period
      sampling = configFile.get<unsigned int>("sampling",sampling); // lire le nombre de pas de temps entre chaque ecriture des diagnostics
      
      // Ouverture du fichier de sortie
      outputFile = new ofstream(configFile.get<string>("output").c_str());
      outputFile->precision(15);
      dt = tf/nsteps_per;
    };


    // Destructeur virtuel
    virtual ~Engine()
    {
      outputFile->close();
      delete outputFile;
    };
      // Simulation complete
    void run()
    {
      t = 0.;
      last = 0;
      printOut(true);

      while( t < tf-0.5*dt )
      {
        step();
        printOut(false);
      }
      printOut(true);
    };
};

// programme
int main(int argc, char* argv[])
{
  // Existing main function implementation
  // ...
  string inputPath("configuration.in.example"); // Fichier d'input par defaut
  if(argc>1) // Fichier d'input specifie par l'utilisateur ("./Exercice2 config_perso.in")
      inputPath = argv[1];

  ConfigFile configFile(inputPath); // Les parametres sont lus et stockes dans une "map" de strings.

  for(int i(2); i<argc; ++i) // Input complementaires ("./Exercice2 config_perso.in input_scan=[valeur]")
      configFile.process(argv[i]);

  Engine* engine;

  // Create an instance of Engine instead of EngineEuler
  engine = new Engine(configFile);

  engine->run(); // executer la simulation

  delete engine; // effacer la class simulation 
  cout << "Fin de la simulation." << endl;
  return 0;
}


