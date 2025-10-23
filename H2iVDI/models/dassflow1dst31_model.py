# import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo

try:
    from dassflow1d.m_mesh import Mesh
    import dassflow1d
    import dassflow1d.m_control as m_control
    import dassflow1d.m_obs as m_obs
    import dassflow1d.m_sw_mono as m_sw_mono
except:
    dassflow1d = None


class DassFlow1DST31BModel:

    def __init__(self, data, h0=1.0, h1=None, k0=25.0, k1=None, bathy_model="h1", kch=None, dx=500):

        if dassflow1d is None:
            raise RuntimeError("DassFlow-1D code is not installed. Please install it or use another core model.")
        self._data = data
        self._h0 = h0
        if h1 is None:
            h1 = np.ones(data.H.shape[1])
        elif isinstance(h1, float) or isinstance(h1, int):
            h1 = np.ones(data.H.shape[1]) * h1
        self._h1 = h1
        self._k0 = k0
        if k1 is None:
            k1 = np.ones(data.H.shape[1])
        elif isinstance(k1, float) or isinstance(k1, int):
            k1 = np.ones(data.H.shape[1]) * k1
        self._k1 = k1
        self._bathy_model = bathy_model
        if kch is None:
            self._kch = None
        if isinstance(kch, float) or isinstance(kch, int):
            self._kch = np.ones(data.H.shape[1]) * kch
        else:
            self._kch = kch
        self._dx = dx

        self._mesh = self._create_mesh_(data)
        self._model = self._create_model_(self._mesh)

    @property
    def name(self):
        return "DassFlow1DST31"

    def solve(self, Qin, data=None):

        if data is None:
            data = self._data
        mesh = self._model.msh
        model = self._model

        # Compute and apply bathymetry
        offset = self._mesh.seg[0].first_cs-1
        if self._bathy_model == "lf":
            href = self._h0
            lowp = np.argmin(self._data.H[:, 0])
            Dref = (self._data.W[lowp, 0] * np.sqrt(self._data.S[lowp, 0]))**(3./5.)
            D = (self._data.W[lowp, :] * np.sqrt(self._data.S[lowp, :]))**(-3./5.)
            h = href * Dref * D
            h0 = np.mean(h0)
            h1 = h / h0
        else:
            h0 = self._h0
            h1 = self._h1
        for r in range(0, self._data.x.size):
            self._mesh.cs[r+offset].bathy = self._mesh.cs[r+offset].level_heights[0] - h0 * h1[r]
        self._mesh.update_geometries()
        self._mesh.update_bathy_slopes()

        # Apply stricklers
        if self._kch is not None:
            for r in range(0, self._data.x.size):
                self._mesh.cs[r+offset].strickler_params[0] = self._k0 * self._k1[r]
                self._mesh.cs[r+offset].strickler_params[1] = self._kch[r]
                self._mesh.cs[r+offset].strickler_params[2] = self._k0 * self._k1[r]
                # if r == 0:
                #     print("K%i:" % r, self._mesh.cs[r+offset].strickler_params[:])
        else:
            for r in range(0, self._data.x.size):
                self._mesh.cs[r+offset].strickler_params[0] = self._k0 * self._k1[r]
                if r == 0:
                    print("(code=%i) K%i:" % (self._mesh.strickler_type_code, r), self._mesh.cs[r+offset].strickler_params[:])


        # Create pseudo observations
        nt = data.H.shape[0]
        tobs = np.linspace(0.0, 86400.0 * nt, nt)
        obs = m_obs.Observations(data.x.size)
        for r in range(0, data.x.size):
            Hobs = data.H[:, r]
            HWobs = np.ones((2, Hobs.size)) * -1e+99
            HWobs[0, :] = Hobs
            obs.stations[r].setup(mesh, tobs, HWobs, indices=r+1)

            
        # Set boundary conditions
        # print("setup BCs")
        model.bc[0].set_timeseries(t=tobs, y=Qin)
        model.bc[1].set_timeseries(t=tobs, y=data.H[:, -1])
        
        # Update end time
        model.te = tobs[-1]

        # print("Create control")
        control = m_control.Control()
        # print("Run and compute cost")
        cost = dassflow1d.calc_cost(model, control, obs)
        H = obs.est[0, :].reshape(data.H.shape[::-1]).T.copy()
        return H

    def cost(self, Qin, data):
        H = self.solve(Qin, data)
        residuals = np.ravel(H - data.H)
        cost = np.sum(residuals[np.isfinite(np.ravel(data.H))]**2)
        if np.isnan(cost):
            cost = 1e+99
        if cost < 1e-18:
            print(H)
            print(data.H)
            choice = input()
        return cost

    def set_data(self, data):
        self._data = data

    def set_h0(self, h0):
        self._h0 = h0
        offset = self._mesh.seg[0].first_cs-1
        if self._bathy_model == "lf":
            href = h0
            lowp = np.argmin(self._data.H[:, 0])
            Dref = (self._data.W[lowp, 0] * np.sqrt(self._data.S[lowp, 0]))**(3./5.)
            D = (self._data.W[lowp, :] * np.sqrt(self._data.S[lowp, :]))**(-3./5.)
            h = href * Dref * D
            h1 = h / np.mean(h)
        else:
            h1 = self._h1
        for r in range(0, self._data.x.size):
            self._mesh.cs[r+offset].bathy = self._mesh.cs[r+offset].level_heights[0] - h0 * h1[r]
        self._mesh.update_geometries()
        self._mesh.update_bathy_slopes()

    def set_h1(self, h1):
        raise RuntimeError("Not implemented yet")
        # if self._bathy_model == "lf":
        #     raise RuntimeError("Cannot set h1 when bathy_model is 'lf'")
        # mesh_x = self._mesh.get_segment_field(0, "x")
        # if h1.size != data.x.size:
        #     raise RuntimeError("'h1' must be of same size as data.x")
        # for r in range(0, self._data.x.size):
        #     self._mesh.cs[r+offset].bathy = self._mesh.cs[r+offset].level_heights[0] - self._h0 * h1[r]
        # self._mesh.update_geometries()
        # self._mesh.update_bathy_slopes()

    def set_h(self, h):
        self._h0 = np.mean(h)
        self._h1 = h / np.mean(h)
        offset = self._mesh.seg[0].first_cs-1
        if self._bathy_model == "lf":
            href = self._h0
            lowp = np.argmin(self._data.H[:, 0])
            Dref = (self._data.W[lowp, 0] * np.sqrt(self._data.S[lowp, 0]))**(3./5.)
            D = (self._data.W[lowp, :] * np.sqrt(self._data.S[lowp, :]))**(-3./5.)
            h = href * Dref * D
            self._h1 = h / np.mean(h)
        else:
            h1 = self._h1
        for r in range(0, self._data.x.size):
            self._mesh.cs[r+offset].bathy = self._mesh.cs[r+offset].level_heights[0] - self._h0 * h1[r]
        self._mesh.update_geometries()
        self._mesh.update_bathy_slopes()

    def set_k0(self, k0):
        self._k0 = k0

    def set_kch(self, kch):
        self._kch = kch

    def compute_lowfroude_discharge(self):
        data = self._data
        h = self._h0 * self._h1
        #b = data.H0r - h
        K = self._k0 * self._k1

        #W = np.repeat(data.Wr.reshape((1, -1)), data.H.shape[0], axis=0)
        A0 = np.repeat((data.We[0, :] * h).reshape((1, -1)), data.H.shape[0], axis=0)
        #A = (data.H - np.repeat(b.reshape((1, -1)), data.H.shape[0], axis=0)) * W
        A = A0 + data._dAr
        K = np.repeat(K.reshape((1, -1)), data.H.shape[0], axis=0)

        phi = K * A**(5./3.) * data._Wr**(-2./3.)

        Qlf = phi * data.S**0.5
        # if np.any(Qlf > 1e+5):
        #     print("Qlf overflow:")
        #     print("- A: %f %f" % (np.min(np.ravel(A)), np.max(np.ravel(A))))
        #     print("- W: %f %f" % (np.min(np.ravel(data.We[0, :])), np.max(np.ravel(data.We[0, :]))))
        #     print("- S: %f %f" % (np.min(np.ravel(data.S)), np.max(np.ravel(data.S))))
        #     plt.plot(data.S)
        #     plt.show()
            # choice = input()

        return Qlf
    
    def compute_optim_qin(self, q0_bounds):

        def minimize_fun(x, model, Qin1):

            Qin = x * Qin1
            return model.cost(Qin, model._data)

        Qlf = self.compute_lowfroude_discharge()
        Qin = np.nanmean(Qlf, axis=1)
        Qin1 = Qin / np.mean(Qin)

        res = spo.minimize_scalar(minimize_fun, bounds=q0_bounds, args=(self, Qin1), options={"maxiter": 10})
        Qin = res.x * Qin1

        return Qin
    

    def _create_mesh_(self, data):
        
        mesh = Mesh(data.x.size, 1)
        for r in range(0, data.x.size):
            mesh.cs[r].set_coords(data.x[r], 0.0)
            mesh.cs[r].x = data.x[r]
            if np.any(np.isnan(data.He[:, r])):
                mesh.cs[r].set_levels(data.H0r[r], data.Wr[r])
                mesh.cs[r].bathy = data.H0r[r] - self._h0 * self._h1[r]
            else:
                mesh.cs[r].set_levels(data.He[:, r], data.We[:, r])
                mesh.cs[r].bathy = data.He[0, r] - self._h0 * self._h1[r]
            mesh.cs[r].strickler_params[0] = self._k0 * self._k1[r]
            

        mesh.setup_segment(0, 0, data.x.size, [-1], [-1])
        mesh.add_ghost_cells()
        mesh.finalise_curvilinear_abscissae()
        mesh.update_geometries()
        mesh.update_bathy_slopes()
        mesh.check()
        print("Mesh has %i cross-sections" % (mesh.seg[0].last_cs - mesh.seg[0].first_cs + 1))
        dassflow1d.write_mesh("mesh.geo", mesh)
        mesh.resample(self._dx)
        print("Mesh has %i cross-sections" % (mesh.seg[0].last_cs - mesh.seg[0].first_cs + 1))
        return mesh

    def _create_model_(self, mesh):

        if self._kch is None:
            mesh.set_strickler_type("constant")
            mesh.set_uniform_strickler_parameters([self._k0])
        else:
            mesh.set_strickler_type("Debord")
            print("K:",[self._k0, self._kch, self._k0])
            mesh.set_uniform_strickler_parameters([self._k0, np.mean(self._kch), self._k0])
        model = m_sw_mono.Model(mesh)
        model.bc[0].id = "discharge"
        model.bc[1].id = "elevation"
        model.set_scheme("preissmann")
        model.ts = 0
        model.te = 0
        model.dt = 10800
        model.dtout = 86400
        model.print_progress = False
        model.disable_stdout = True
        model.heps = 0.1
        model.steady_states = True
        self._model = model
        self._mesh = self._model.msh
        return model    