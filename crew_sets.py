import pandas

# a set of unique ids of pilots
all_pilots = {'900201','900488', '900421', '900424'}

# flight names
flight_names = {'a320', 'a330'}


# #ranks, bases and flights are dictionaries 

ranks = {

'cpt' : {'900201','900488', '900421'},
'fo' : {'900424'}

}

bases = {

'b1' : {'900488', '900421'},
'b2' : {'900201', '900424'}

}

flights = {

'a330' : {'900201', '900421'},
'a320' : {'900488', '900424'}

}

## "from" and "to" dictionaries, should have 3*2 = 6 dictionaries

from_flights = {
    'a330' : {'900201'},
    'a320' : {'900488'}
}

to_flights = {
    'a330' : {'900488'},
    'a320' : {'900201'}
}

##
for flight,pilots in flights.iteritems() :
     print flight, pilots
     