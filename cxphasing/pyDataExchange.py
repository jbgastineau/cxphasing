import h5py

class DataExchange(object):

	def __init__(self):
		pass

	def _init_file(self, filename):
		self.f = h5py.File(filename, 'w')
		self._groups = {}
		self._datasets = {}

		self.implements = self.f.create_dataset('implements', data = "")

		self.exchange = self.f.create_group('exchange')



	def add_group(self, root, group_name):
		# add group_name under root
		self.implements = self.implements.data+group_name
		self._groups[group_name] = root.create_dataset(group_name)

	def add_dataset(self, root, dataset_name, dataset, units = 'Undefined'):
		self._datasets[dataset_name] = root.create_dataset(dataset_name, data=dataset)
		self._datasets[dataset_name].attrs['units'] = units
