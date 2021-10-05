import time
import napari
import numpy as np

import cestimii.geomshapes as gs
import cestimii.curvature as cv

print("Let's begin. Next step is the initialisation of the testshape.")
starttime = time.time()

ocg = gs.testshape_ocg()  # initialise occupancy grid of "testshape"
print("Initialised occupancy grid of the testshape. Next up are curvature estimations.\n"
      + "This might take a few minutes on slow machines.\n"
      + "Current Runtime: " + str(time.time() - starttime))

# calculate (principal) curvature estimations of "testshape"
cest3 = cv.cepca_ocg(ocg, kr=3)
print("Calculated principal curvature estimations and eigenvectors on scale 3.\n"
      + "Current Runtime: " + str(time.time() - starttime))
cest12 = cv.cepca_ocg(ocg, kr=12)
print("Calculated principal curvature estimations and eigenvectors on scale 12.\n"
      + "Current Runtime: " + str(time.time() - starttime))
cest36 = cv.cepca_ocg(ocg, kr=36)
print("Calculated principal curvature estimations and eigenvectors on scale 36.\n"
      + "Current Runtime: " + str(time.time() - starttime))
cestavg = cv.cepca_msavg_ocg(ocg, startscale=9, endscale=17, scaledist=2)
print("Calculated principal curvature estimations and eigenvectors on a scales 9,11,...,17. Next: mean curvatures.\n"
      + "Current Runtime: " + str(time.time() - starttime))

# calculate mean curvature of "testshape"
mean3 = cv.cemean_principalcurv(cest3[0], cest3[1])
mean12 = cv.cemean_principalcurv(cest12[0], cest12[1])
mean36 = cv.cemean_principalcurv(cest36[0], cest36[1])
meanavg = cv.cemean_principalcurv(cestavg[0], cestavg[1])
print("Calculated mean curvatures from the previous estimations.\n"
      + "Current Runtime: " + str(time.time() - starttime))

# visualise mean curvatures in napari (value range is shifted to allow better visualisations)
viewer = napari.Viewer()  # initialise viewer object
viewer.add_image((mean3 - np.min(mean3)) * ocg, rgb=False, colormap='viridis', rendering='attenuated_mip',
                 interpolation='nearest', attenuation=0.5, name='r=3, mean curvature', visible=False)
viewer.add_image((mean12 - np.min(mean12)) * ocg, rgb=False, colormap='viridis', rendering='attenuated_mip',
                 interpolation='nearest', attenuation=0.5, name='r=12, mean curvature', visible=False)
viewer.add_image((mean36 - np.min(mean36)) * ocg, rgb=False, colormap='viridis', rendering='attenuated_mip',
                 interpolation='nearest', attenuation=0.5, name='r=36, mean curvature', visible=False)
viewer.add_image((meanavg - np.min(meanavg)) * ocg, rgb=False, colormap='viridis', rendering='attenuated_mip',
                 interpolation='nearest', attenuation=0.5, name='r=9,11,...,17, avg mean curvature', visible=False)
print("Finished visualisation in napari. Examine the newly opened tab. Enjoy.\n"
      + "Current Runtime: " + str(time.time() - starttime))

input("Press anything to exit the script and close napari...")
