#!/usr/bin/env python

import lsst.daf.butler as dafButler

butler = dafButler.Butler('/repo/LSSTComCam', collections='u/lguy/LVV-T191')

print(f"Querying data products for test LVV-T191")

registry = butler.registry
datasetTypes = registry.queryDatasetTypes()
print(f"Dataset types:i\n{datasetTypes}")


datasetType = 'calexp'
dataId = {'visit': 2024102700018, 'detector': 5}
datasetRefs = butler.query_datasets(datasetType, data_id=dataId)
calexp = butler.get(datasetType,  dataId=dataId)

visitInfo = calexp.visitInfo
print(visitInfo)
print(calexp.wcs)
