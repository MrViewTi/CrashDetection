nn(dangerdist_cnn, [X]) :: car_obstruction_distance(X).
nn(dangerordr_cnn, [X]) :: car_obstruction_order(X)

crash(X) :- car_obstruction_distance(X), car_obstruction_order(X).