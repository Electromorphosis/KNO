# Raport z laboratorium nr 3 - Klasyfikacja

## 1. Opis modeli
Do porównania zaimplementowane zostały dwa modele sekwencyjnych sieci neuronowych, z obiektu 
```tf.keras.Sequential```. Oba miały zbliżóną budowę, jednak drugi posiadał więcej neuronów na warstwach - dzięki temu za pomocą eksperymentu możliwe było sprawdzenie czy większa liczba neuronóœ poprawi dokładność predykcji.

### Model "Standard"
1. Warstwa wejść - 12 neuronów, aktywacja "relu".
2. Warstwa dropout z wartością parametru 0.2.
3. Warstwa typu "dense" z 12 neuronami i aktywacją 'relu'.
4. Warstwa wyjściowa, 3 neurony, aktywacja typu "softmax".

### Model "Powiększony (Bigger)" 
1. Warstwa wejściowa, 24 neurony, aktywacja "relu".
2. Warstwa dropout - j/w.
3. Warstwa typu "dense", 24 neurony, "relu".
4. Warstwa wyjściowa, j/w.

## 2. Wyniki

### Model "Standard"
![standard_plots.png](standard_plots.png)

Końcowa dokładność modelu wyniosła: 0.9355

### Model "Powiększony"
![bigger_plots.png](bigger_plots.png)

Końcowa dokładność modelu wyniosła: 0.5511

## 3. Dyskusja
Podczas opracowywania modeli dużym problemem okazało się uzyskanie powtarzalnych wyników pod względem wyjściowej dokładności pomiędzy sesjami treningowymi. W tym wypadku zaobserwowano lepszą powtarzalność wyników po ustawieniu stałych wartości ziarna dla generatorów np. podziału na grupy _train/test_ oraz dodanie warstwy Dropout.

Drugim zaobserwowanym problemem było uzyskanie powtarzalnej poprawy dokładności modelu z biegiem czasu. W tym wypadku wyniki znaczącą się poprawiły po lepszym dostosowaniu parametrów modelu (dokładnie ustawieniu typu loss na 'categorical_crossentropy'), normalizacji danych za pomocą scaler.fit_transform z obiektu StandardScaler pakietu sklearn oraz dodania parametru callback ustawionego na zatrzymywanie procesu uczenia jeśli przez więcej niż zadaną liczbę epok nie następuje poprawa dokładności.

Po tego rodzaju dostosowaniach, model Standard był w stanie przejść przez wszystkie 20 epok, realizując "książkową" krzywą wzrostu dokładności modelu i zadowalającą jak na prosty eksperyment końcową jej wartość na poziomie ok. 0.8. W przypadku modelu "Powiększonego" można z kolei zaobserwować tendencję do dużo skromniejszego wzrostu dokładności, który dodatkowo przy ustawionym poziomie cierpliwości na 4 epoki spowodował szybkie zakończenie eksperymentu z niezadowalającym wynikiem, pomimo ogólnie wzrostowej tendencji dokładności na danych walidacyjnych. Z powodu wspomnianego wzrostu dokładności inferencji na danych walidacyjnych, zdecydowano się na ponowienie procesu trenowania z wyłączonym parametrem callback.

> ### Model "Powiększony" - bez callback
> ![bigger_no_callback_plots.png](bigger_no_callback_plots.png)
>
> Końcowa dokładność: 0.8372

Na podstawie tego i kilku dodatkowych, nieopisanych tutaj szczegółowo powtórzeń uruchomienia programu można zauważyć, że po wyłączeniu przedterminowego abortowania treningu, większy model był w stanie osiągnąć dużo lepsze wyniki, porównywalne do mniejszego modelu. Wydają się wypływać z tego dwa wnioski:
- Większy model charakteryzowała większa chaotyczność (entropia?) procesu uczenia się - model ma większą "skłonność" do niepoprawiania lub nawet pogarszania swojej dokładności pomiędzy rundami treningowymi.
- Uśredniając większy model cechował się nieco gorszą dokładnością po takiej samej liczbie epok treningowych niż mniejszy - to istotna informacja, która sprawia, że to mniejszy model jednoznacznie wydaje się być skuteczniejszy w przypadku danych o takiej wymiarowości (lub innych cechach), będąc przy tym oszczędniejszy obliczeniowo. 

Wniosek z punktu drugiego można skrótowo podsumować:
> _Większy (model) nie zawsze znaczy lepszy_