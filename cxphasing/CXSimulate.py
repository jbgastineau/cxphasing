import cxphasing.cxparams.CXParams as CXP
import cxphasing.CXPhasing as CXPh
import cxphasing.CXData as CXData
from data_exchange import DataExchangeFile, DataExchangeEntry
import scipy as sp
import pdb

def pack_data_exchange():
    f = DataExchangeFile(CXP.io.data_exchange_filename, mode='w')
    sim = DataExchangeEntry.simulation(
                                        name={'value': 'Simulated Ptycho Data.'},
                                        energy={'value': CXP.experiment.energy, 'units':'keV'},
                                        focusing_optic={'value': CXP.experiment.optic},
                                        probe_modes={'value':CXP.reconstruction.probe_modes},
                                        noise_model={'value': CXP.simulation.noise_model},
                                        gaussian_noise_level={'value': CXP.simulation.gaussian_noise_level},
                                        total_photons={'value': CXP.simulation.total_photons},
                                        beam_stop={'value': CXP.simulation.beam_stop},
                                        beam_stop_size={'value':CXP.simulation.beam_stop_size},
                                        beam_stop_attenuation={'value':CXP.simulation.beam_stop_attenuation},
                                        defocus = {'value':CXP.simulation.defocus},
                                        position_jitter={'value': CXP.reconstruction.initial_position_jitter_radius, 'units':'pixels'}
        )
    f.add_entry(sim)
    sample = DataExchangeEntry.sample(
                                            root='/simulation',
                                            name={'value':'ground truth sample complex amplitude'}, 
                                            data={'value': cxph.sample.data[0], 'units':'sqrt(counts)'},
        )
    f.add_entry(sample)
    probe = DataExchangeEntry.sample(
                                            root='/simulation',
                                            entry_name='probe',
        )
    for mode in range(CXP.reconstruction.probe_modes):
        setattr(probe, 'mode_{:d}'.format(mode), {'value': cxph.input_probe.modes[mode].data[0], 'units':'counts'})

    f.add_entry(probe)
    detector = DataExchangeEntry.detector(
                                            root='/simulation',
                                            x_pixel_size={'value': CXP.experiment.dx_d},
                                            y_pixel_size={'value': CXP.experiment.dx_d},
                                            x_dimension={'value': CXP.experiment.px},
                                            y_dimension={'value': CXP.experiment.py},
                                            distance={'value': CXP.experiment.z},
                                            basis_vectors={'value': [[0,-CXP.experiment.dx_d,0],[-CXP.experiment.dx_d,0,0]]},
                                            corner_position={'value': [0,0,0]}
        )
    f.add_entry(detector)
    data = DataExchangeEntry.data(
                                            name={'value': 'simulated_data'},
                                            data={'value': sp.array(cxph.det_mod.data), 'axes':'translation:y:x', 'units':'counts',
                                            'dataset_opts':  {'compression': 'gzip', 'compression_opts': 4}},
                                            translation={'value':'/exchange/sample/geometry/translation'}          
                                )
    f.add_entry(data)
    # Get scan positions into dex format
    pos = sp.zeros((cxph.positions.total, 3))
    y, x = cxph.positions.correct
    for i in range(cxph.positions.total):
        pos[i,0], pos[i, 1] = x[i]*CXP.dx_s, y[i]*CXP.dx_s

    positions = DataExchangeEntry.translation(
                                            root='/exchange/sample/geometry',
                                            name={'value':'ptychography scan positions'},
                                            scan_type={'value': CXP.measurement.ptycho_scan_type},
                                            data={'value': pos, 'units': 'm'}
                                )

    
    f.add_entry(positions)
    f.close()


if __name__=='__main__':

    cxph = CXPh()
    cxph.positions = CXData(name='positions', data=[])
    cxph.ptycho_mesh()
    cxph.simulate_data(no_save=True)
    pack_data_exchange()