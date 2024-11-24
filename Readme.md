# KNO Common Repository

## KNO Lab 4
- [x] Podziel zbiór danych na 3 podzbiory - treningowy, walidacyjny i testowy. Upewnij się, że za każdym razem ten podział będzie taki sam. Zachęcam do poproszenia prowadzącego w celu konsultacji.
- [x] Zapisz wyniki Twojego modelu z poprzednich zajęć na zbiorze walidacyjnym i testowym. Będzie to tzw "baseline", czyli punkt odniesienia, do którego będziemy porównywali. Zakładamy, że uda się poprawić wyniki. 
- [x] Uporządkuj kod tak, aby tworzenie modelu było oddzielną funkcją. Funkcja ta powinna przyjmować parametry, a zwracać model. Zobacz przykład załączony na dole. Istotna uwaga - niech będą dwie sieci o różnym układzie warstw (wielkość/liczba warstw).
- [x] Zdecyduj się na zestaw 3 parametrów które będziesz optymalizować. Wymagane to tempo uczenia i sieć neuronowa (czyli różne układy warstw), a trzeci parametr proszę wybrać. Parametry mogą być różnych typów.
- [x] Wykonaj eksperyment dla wszystkich kombinacji wartości parametrów (2 możliwości per parametr) - nauka na zbiorze treningowym a walidacja na walidacyjnym.
- [ ] Zobacz, jaki zestaw parametrów jest najlepszy. Do tego celu wykorzystaj wyniki na zbiorze walidacyjnym (już je masz)
- [ ] Przetestuj model na zbiorze testowym. Porównaj z wynikiem dla modelu bazowego (baseline ).
- [ ] Zautomatyzuj cały proces, to znaczy - przygotuj skrypt który automatycznie przetestuje dla wszystkich kombinacji parametrów wyniki dla Twoich modeli i wygeneruje wyniki w postaci tabeli + wskaże najlepszy model.
- [ ] Korzystając z TensorBoard stwórz wizualizację Twojego modelu sieci neuronowej.