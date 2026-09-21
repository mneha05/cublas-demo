pipeline {
  agent { label 'nvidia-gpu' }
  stages {
    stage('Configure') { steps { sh 'cmake -S . -B build' } }
    stage('Build') { steps { sh 'cmake --build build -j' } }
    stage('Run') { steps { sh './build/cuda_linear_bench' } }
  }
}
