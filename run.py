import sys, os
from ldmtools import *
import imageio
import numpy as np
from multiprocessing import Pool
from VotingTreeRegressor import VotingTreeRegressor
from neubiaswg5 import CLASS_LNDDET
from neubiaswg5.helpers import NeubiasJob, prepare_data, get_discipline
from cytomine.models import Annotation, Job, ImageInstanceCollection, AnnotationCollection, Property, AttachedFileCollection, AttachedFile
import joblib

def	get_neubias_coords(gt_path, tr_im):
	first_im = imageio.imread(os.path.join(gt_path, '%d.tif'%tr_im[0]))
	nldms = np.max(first_im)
	nimages = len(tr_im)
	xcs = np.zeros((nimages, nldms))
	ycs = np.zeros((nimages, nldms))
	xrs = np.zeros((nimages, nldms))
	yrs = np.zeros((nimages, nldms))
	for i in range(len(tr_im)):
		id = tr_im[i]
		gt_img = imageio.imread(os.path.join(gt_path, '%d.tif'%id))
		for id_term in range(1, nldms+1):
			(y, x) = np.where(gt_img==id_term)
			(h, w) = gt_img.shape
			yc = y[0]
			xc = x[0]
			yr = yc/h
			xr = xc/w
			xcs[i, id_term-1] = xc
			ycs[i, id_term-1] = yc
			xrs[i, id_term-1] = xr
			yrs[i, id_term-1] = yr
	return np.array(xcs), np.array(ycs), np.array(xrs), np.array(yrs)

if __name__ == "__main__":
	with NeubiasJob.from_cli(sys.argv) as conn:
		problem_cls = get_discipline(conn, default=CLASS_LNDDET)
		conn.job.update(status=Job.RUNNING, statusComment="Initialization of the training phase")
		in_images, gt_images, in_path, gt_path, out_path, tmp_path = prepare_data(problem_cls, conn, is_2d=True, **conn.flags)
		tmax = 1
		for f in os.listdir(gt_path):
			if f.endswith('.tif'):
				gt_img = imageio.imread(os.path.join(gt_path, f))
				tmax = np.max(gt_img)
				break

		term_list = range(1, tmax+1)
		tr_im = [int(id_im) for id_im in conn.parameters.cytomine_training_images.split(',')]
		DATA = None
		REP = None
		be = 0
		sfinal = ""
		for id_term in term_list:
			sfinal += "%d " % id_term
		sfinal = sfinal.rstrip(' ')

		#job_parameters = {}
		#job_parameters['cytomine_id_terms'] = sfinal.replace(' ', ',')
		#job_parameters['model_njobs'] = conn.parameters.model_njobs
		#job_parameters['model_D_MAX'] = conn.parameters.model_D_MAX
		#job_parameters['model_W'] = conn.parameters.model_W
		#job_parameters['model_T'] = conn.parameters.model_T
		#job_parameters['model_n_samples'] = conn.parameters.model_n_samples
		#job_parameters['model_n_reduc'] = conn.parameters.model_n_reduc
		#job_parameters['model_n'] = conn.parameters.model_n

		#DROPPED PARAMETERS BECAUSE THEY ARE NOT USED AT TRAINING
		#DON'T FORGET TO ADD THEM AT PREDICTION...
		#job_parameters['model_R_MAX'] = conn.parameters.model_R_MAX
		#job_parameters['model_R_MIN'] = conn.parameters.model_R_MIN
		#job_parameters['model_alpha'] = conn.parameters.model_alpha
		#job_parameters['model_step'] = conn.parameters.model_step

		(xc, yc, xr, yr) = get_neubias_coords(gt_path, tr_im)
		(nims, nldms) = xc.shape
		Xc = np.zeros((nims, len(term_list)))
		Yc = np.zeros(Xc.shape)
		i = 0
		for id_term in term_list:
			Xc[:, id_term-1] = xc[:, id_term-1]
			Yc[:, id_term-1] = yc[:, id_term-1]

		(nims,nldms) = Xc.shape
		im_list = tr_im

		h2 = generate_2_horizontal(conn.parameters.model_W, conn.parameters.model_n)
		v2 = generate_2_vertical(conn.parameters.model_W, conn.parameters.model_n)
		h3 = generate_3_horizontal(conn.parameters.model_W, conn.parameters.model_n)
		v3 = generate_3_vertical(conn.parameters.model_W, conn.parameters.model_n)
		sq = generate_square(conn.parameters.model_W, conn.parameters.model_n)

		for id_term in conn.monitor(term_list, start=10, end=80, period=0.05, prefix="Visual model building for terms..."):
			(dataset,rep,img) = build_dataset_image_offset_mp(in_path, Xc[:, id_term-1], Yc[:, id_term-1], im_list, conn.parameters.model_D_MAX, conn.parameters.model_n_samples, h2, v2, h3, v3, sq, conn.parameters.model_njobs)
			clf = VotingTreeRegressor(n_estimators=conn.parameters.model_T,n_jobs=conn.parameters.model_njobs)
			clf = clf.fit(dataset,rep)
			model_filename = joblib.dump(clf, os.path.join(out_path, '%d_model.joblib' % (id_term)), compress=3)[0]
			AttachedFile(
				conn.job,
				domainIdent=conn.job.id,
				filename=model_filename,
				domainClassName="be.cytomine.processing.Job"
			).upload()

		conn.job.update(status=Job.RUNNING, progress=80, statusComment="Computing the post-processing model...")
		xt = procrustes(Xc,Yc)
		(mu,P) = apply_pca(xt,conn.parameters.model_n_reduc)
		muP_filename = joblib.dump((mu,P),'muP.joblib', compress=3)[0]
		features_filename = joblib.dump((h2, v2, h3, v3, sq), 'features.joblib', compress=3)[0]
		AttachedFile(
			conn.job,
			domainIdent=conn.job.id,
			filename=muP_filename,
			domainClassName="be.cytomine.processing.Job"
		).upload()
		AttachedFile(
			conn.job,
			domainIdent=conn.job.id,
			filename=features_filename,
			domainClassName="be.cytomine.processing.Job"
		).upload()
		Property(conn.job, key="id_terms", value=sfinal.rstrip(" ")).save()
		conn.job.update(progress=100, status=Job.TERMINATED, statusComment="Job terminated.")