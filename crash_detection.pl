nn(dangerdist_cnn, [X]) :: car_obstruction_distance(X).

crash(X) :- car_obstruction_distance(X).