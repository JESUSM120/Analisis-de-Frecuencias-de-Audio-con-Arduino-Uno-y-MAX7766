#include <arduinoFFT.h>

#define SAMPLES 128               // Debe ser potencia de 2
#define SAMPLING_FREQUENCY 8000   // Hz

arduinoFFT FFT = arduinoFFT();

unsigned int sampling_period_us;
unsigned long microseconds;
double vReal[SAMPLES];
double vImag[SAMPLES];

void setup() {
  Serial.begin(115200);
  sampling_period_us = round(1000000.0 / SAMPLING_FREQUENCY);

  while (!Serial) { ; }

  // Config para que Python se auto-ajuste
  Serial.print("CONFIG:");
  Serial.print(SAMPLING_FREQUENCY);
  Serial.print(",");
  Serial.println(SAMPLES);
}

void loop() {
  double sum = 0;

  // 1. MUESTREO
  for (int i = 0; i < SAMPLES; i++) {
    microseconds = micros();
    vReal[i] = analogRead(A0);
    sum += vReal[i];
    vImag[i] = 0;
    while (micros() - microseconds < sampling_period_us) {
      // Mantener frecuencia de muestreo
    }
  }

  // 2. CÁLCULO DE OFFSET Y CENTRADO
  double mean = sum / SAMPLES;  // Offset real
  double target = 512.0;        // 512 ≈ 2.5 V en ADC de 10 bits (0-1023)
  double shift = target - mean; // Desplazamiento necesario
  for (int i = 0; i < SAMPLES; i++) {
    vReal[i] += shift;          // Mover toda la onda hacia el centro 2.5 V
    if (vReal[i] < 0) vReal[i] = 0;       // Limitar a rango ADC
    if (vReal[i] > 1023) vReal[i] = 1023;
  }

  // 3. ENVIAR ONDA AJUSTADA
  Serial.print("WAV:");
  for (int i = 0; i < SAMPLES; i++) {
    Serial.print(vReal[i]);
    if (i < SAMPLES - 1) Serial.print(",");
  }
  Serial.println();

  // 4. FFT
  double temp[SAMPLES];
  for (int i = 0; i < SAMPLES; i++) temp[i] = vReal[i] - target; // Quitar DC antes de FFT

  FFT.Windowing(temp, SAMPLES, FFT_WIN_TYP_HAMMING, FFT_FORWARD);
  FFT.Compute(temp, vImag, SAMPLES, FFT_FORWARD);
  FFT.ComplexToMagnitude(temp, vImag, SAMPLES);

  Serial.print("FFT:");
  for (int i = 1; i < (SAMPLES / 2); i++) {
    Serial.print(temp[i]);
    if (i < (SAMPLES / 2) - 1) Serial.print(",");
  }
  Serial.println();
}


