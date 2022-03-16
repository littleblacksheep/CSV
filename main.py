from optparse import OptionParser
import utils as u
import train as t
import analyze as a
import dataloader as d
import SPH.generate as g

if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option('--config', type=str, dest='config', default='./config.yaml')
    (args, _) = parser.parse_args()
    cfg = u.system.prepare_cfg(args)

    if cfg["system"]["mode"] == "si":
        g.generate(cfg)
    elif cfg["system"]["mode"] == "ad":
        a.analyze_data(cfg)
    elif cfg["system"]["mode"] == "tr":
        t.train(cfg)
    elif cfg["system"]["mode"] == "dr":
        u.ML.download_results()
    elif cfg["system"]["mode"] == "ar":
        a.analyze_results(cfg)
    elif cfg["system"]["mode"] == "am":
        a.analyze_model(cfg)

    print('INFO: Done.')