from temprature import predictionTemprature
predicted_temprature = predictionTemprature()

from pressure import predictionPressure
predicted_pressure = predictionPressure()

from humidity import predictionHumidity
predicted_humidity = predictionHumidity()

from wind import predictionWind
predicted_wind = predictionWind()

print("Time               TEMPRATURE    PRESSURE    HUMIDITY     WIND SPEED")
d = 9;
m = 7;
y = 2010;
h = 14;
mn = 0;
for i in range(0,864):
    if (mn % 60 == 0):
        if (mn != 0):
            h += 1;
            mn = 0;
    if (h % 24 == 0):
        if(h != 0):
            d += 1;
            h = 0;
    if (d == 31):
        d = 1;
        m += 1;
    if (m == 13):
        m = 1;
    #if(h == 0):
    #if(mn == 0):
    print(str(d) + "-" + str(m) + "-"+ str(y)+" "+ str(h)+":"+str(mn)+"      "+
          str(predicted_temprature[i][0]) + "     " +
          str(predicted_pressure[i][0]) + "     " +
          str(predicted_humidity[i][0]) + "     " +
          str(predicted_wind[i][0]) + "     " );
    mn += 10;


    