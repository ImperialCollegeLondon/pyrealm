
## Temperature modulation of kphio



```{code-cell} python
fk_c3 = pmodel.calc_ftemp_kphio(tc_1d)
fk_c4 = pmodel.calc_ftemp_kphio(tc_1d, c4=True)
pyplot.plot(tc_1d, fk_c3)
pyplot.plot(tc_1d, fk_c4)
pyplot.xlabel('Temperature (°C)')
pyplot.ylabel('ftemp_kphio ')
pyplot.show()
```


## Soil Moisture Stress



```{code-cell} python
soilm_1d = np.linspace(0, 1, n_pts)
meanalpha_1d = np.linspace(0, 1, n_pts)
soilm_2d = np.broadcast_to(soilm_1d, (n_pts, n_pts))
meanalpha_2d = np.broadcast_to(meanalpha_1d, (n_pts, n_pts))

ca = pmodel.calc_soilmstress(soilm_2d, meanalpha_2d.transpose())

fig, ax = pyplot.subplots()
CS = ax.contour(soilm_1d, meanalpha_1d, ca, colors='black', 
                levels=[0.2,0.4,0.6,0.8,0.9,0.95,0.99, 0.999, 1.0])
ax.clabel(CS, inline=1, fontsize=10)
ax.set_title('Soil Moisture Stress')
ax.set_xlabel('Soil moisture')
ax.set_ylabel('Mean alpha')

pyplot.show()
```


```{eval-rst}
.. autoclass:: pyrealm.pmodel.CalcLUEVcmax
    :show-inheritance:
```


```{eval-rst}
.. autofunction:: pyrealm.pmodel.calc_co2_to_ca
```