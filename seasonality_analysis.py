# Found no seasonality in Indonesia's industries over the past few years
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
from statsmodels.graphics.tsaplots import plot_acf

# # Test
# # Generate index values from 0 to 100
# index_values = np.arange(1001)
#
# # Calculate the sine of the index values
# column_values = np.sin(index_values/10)
#
# # Create the pandas Series
# sin_series = pd.Series(column_values, index=index_values)
# figure, axis = plt.subplots(figsize=(10,6))
# plot_acf(sin_series, lags=50, ax=axis)
# plt.show()

# industries = {
#     'basic_materials': ['ADMG.JK', 'AGII.JK', 'AKPI.JK', 'ALDO.JK', 'ALKA.JK', 'ALMI.JK', 'AMMN.JK', 'ANTM.JK',
#                         'APLI.JK', 'ARCI.JK', 'AVIA.JK', 'AYLS.JK', 'BAJA.JK', 'BEBS.JK', 'BMSR.JK', 'BRMS.JK',
#                         'BRNA.JK', 'BRPT.JK', 'BTON.JK', 'CHEM.JK', 'CITA.JK', 'CLPI.JK', 'CMNT.JK', 'CTBN.JK',
#                         'DKFT.JK', 'DPNS.JK', 'EKAD.JK', 'EPAC.JK', 'ESIP.JK', 'ESSA.JK', 'ETWA.JK', 'FASW.JK',
#                         'FPNI.JK', 'FWCT.JK', 'GDST.JK', 'GGRP.JK', 'HKMU.JK', 'IFII.JK', 'IFSH.JK', 'IGAR.JK',
#                         'INAI.JK', 'INCF.JK', 'INCI.JK', 'INCO.JK', 'INKP.JK', 'INRU.JK', 'INTD.JK', 'INTP.JK',
#                         'IPOL.JK', 'ISSP.JK', 'JKSW.JK', 'KAYU.JK', 'KBRI.JK', 'KDSI.JK', 'KKES.JK', 'KMTR.JK',
#                         'KRAS.JK', 'LMSH.JK', 'LTLS.JK', 'MBMA.JK', 'MDKA.JK', 'MDKI.JK', 'MOLI.JK', 'NCKL.JK',
#                         'NICE.JK', 'NICL.JK', 'NIKL.JK', 'NPGF.JK', 'OBMD.JK', 'OKAS.JK', 'OPMS.JK', 'PACK.JK',
#                         'PBID.JK', 'PDPP.JK', 'PICO.JK', 'PNGO.JK', 'PPRI.JK', 'PSAB.JK', 'PURE.JK', 'SAMF.JK',
#                         'SBMA.JK', 'SIMA.JK', 'SMBR.JK', 'SMCB.JK', 'SMGA.JK', 'SMGR.JK', 'SMKL.JK', 'SMLE.JK',
#                         'SPMA.JK', 'SQMI.JK', 'SRSN.JK', 'SULI.JK', 'SWAT.JK', 'TALF.JK', 'TBMS.JK', 'TDPM.JK',
#                         'TINS.JK', 'TIRT.JK', 'TKIM.JK', 'TPIA.JK', 'TRST.JK', 'UNIC.JK', 'WSBP.JK', 'WTON.JK',
#                         'YPAS.JK', 'ZINC.JK'],
#     'consumer_cyclicals': ['ABBA.JK', 'ACES.JK', 'ACRO.JK', 'AEGS.JK', 'AKKU.JK', 'ARGO.JK', 'ARTA.JK', 'ASLC.JK',
#                            'AUTO.JK', 'BABY.JK', 'BAIK.JK', 'BATA.JK', 'BAUT.JK', 'BAYU.JK', 'BELL.JK', 'BIKE.JK',
#                            'BIMA.JK', 'BLTZ.JK', 'BMBL.JK', 'BMTR.JK', 'BOGA.JK', 'BOLA.JK', 'BOLT.JK', 'BRAM.JK',
#                            'BUVA.JK', 'CARS.JK', 'CBMF.JK', 'CINT.JK', 'CLAY.JK', 'CNMA.JK', 'CNTX.JK', 'CSAP.JK',
#                            'CSMI.JK', 'DEPO.JK', 'DFAM.JK', 'DIGI.JK', 'DOOH.JK', 'DRMA.JK', 'DUCK.JK', 'EAST.JK',
#                            'ECII.JK', 'ENAK.JK', 'ERAA.JK', 'ERAL.JK', 'ERTX.JK', 'ESTA.JK', 'ESTI.JK', 'FAST.JK',
#                            'FILM.JK', 'FITT.JK', 'FORU.JK', 'FUTR.JK', 'GDYR.JK', 'GEMA.JK', 'GJTL.JK', 'GLOB.JK',
#                            'GRPH.JK', 'GWSA.JK', 'HAJJ.JK', 'HDTX.JK', 'HOME.JK', 'HOTL.JK', 'HRME.JK', 'HRTA.JK',
#                            'IDEA.JK', 'IIKP.JK', 'IMAS.JK', 'INDR.JK', 'INDS.JK', 'INOV.JK', 'IPTV.JK', 'ISAP.JK',
#                            'JGLE.JK', 'JIHD.JK', 'JSPT.JK', 'KDTN.JK', 'KICI.JK', 'KLIN.JK', 'KOTA.JK', 'KPIG.JK',
#                            'LFLO.JK', 'LIVE.JK', 'LMAX.JK', 'LMPI.JK', 'LPIN.JK', 'LPPF.JK', 'LUCY.JK', 'MABA.JK',
#                            'MAMI.JK', 'MAPA.JK', 'MAPB.JK', 'MAPI.JK', 'MARI.JK', 'MASA.JK', 'MDIA.JK', 'MEJA.JK',
#                            'MGLV.JK', 'MICE.JK', 'MINA.JK', 'MKNT.JK', 'MNCN.JK', 'MPMX.JK', 'MSIN.JK', 'MSKY.JK',
#                            'MYTX.JK', 'NATO.JK', 'NETV.JK', 'NIPS.JK', 'NUSA.JK', 'OLIV.JK', 'PANR.JK', 'PBRX.JK',
#                            'PDES.JK', 'PGLI.JK', 'PJAA.JK', 'PLAN.JK', 'PMJS.JK', 'PNSE.JK', 'POLU.JK', 'POLY.JK',
#                            'PRAS.JK', 'PSKT.JK', 'PTSP.JK', 'PZZA.JK', 'RAAM.JK', 'RAFI.JK', 'RALS.JK', 'RICY.JK',
#                            'SBAT.JK', 'SCMA.JK', 'SCNP.JK', 'SHID.JK', 'SLIS.JK', 'SMSM.JK', 'SNLK.JK', 'SOFA.JK',
#                            'SONA.JK', 'SOTS.JK', 'SRIL.JK', 'SSTM.JK', 'TELE.JK', 'TFCO.JK', 'TMPO.JK', 'TOOL.JK',
#                            'TOYS.JK', 'TRIO.JK', 'TRIS.JK', 'TYRE.JK', 'UFOE.JK', 'UNIT.JK', 'UNTD.JK', 'VIVA.JK',
#                            'VKTR.JK', 'WOOD.JK', 'YELO.JK', 'ZATA.JK', 'ZONE.JK'],
#     'consumer_non_cyclicals': ['AALI.JK', 'ADES.JK', 'AGAR.JK', 'AISA.JK', 'ALTO.JK', 'AMMS.JK', 'AMRT.JK', 'ANDI.JK',
#                                'ANJT.JK', 'ASHA.JK', 'AYAM.JK', 'BEEF.JK', 'BEER.JK', 'BISI.JK', 'BOBA.JK', 'BTEK.JK',
#                                'BUAH.JK', 'BUDI.JK', 'BWPT.JK', 'CAMP.JK', 'CBUT.JK', 'CEKA.JK', 'CLEO.JK', 'CMRY.JK',
#                                'COCO.JK', 'CPIN.JK', 'CPRO.JK', 'CRAB.JK', 'CSRA.JK', 'DAYA.JK', 'DEWI.JK', 'DLTA.JK',
#                                'DMND.JK', 'DPUM.JK', 'DSFI.JK', 'DSNG.JK', 'ENZO.JK', 'EPMT.JK', 'EURO.JK', 'FAPA.JK',
#                                'FISH.JK', 'FLMC.JK', 'FOOD.JK', 'GGRM.JK', 'GOLL.JK', 'GOOD.JK', 'GRPM.JK', 'GULA.JK',
#                                'GZCO.JK', 'HERO.JK', 'HMSP.JK', 'HOKI.JK', 'IBOS.JK', 'ICBP.JK', 'IKAN.JK', 'INDF.JK',
#                                'IPPE.JK', 'ITIC.JK', 'JARR.JK', 'JAWA.JK', 'JPFA.JK', 'KEJU.JK', 'KINO.JK', 'KMDS.JK',
#                                'KPAS.JK', 'LSIP.JK', 'MAGP.JK', 'MAIN.JK', 'MAXI.JK', 'MBTO.JK', 'MGRO.JK', 'MIDI.JK',
#                                'MKTR.JK', 'MLBI.JK', 'MLPL.JK', 'MPPA.JK', 'MRAT.JK', 'MSJA.JK', 'MYOR.JK', 'NANO.JK',
#                                'NASI.JK', 'NAYZ.JK', 'NSSS.JK', 'OILS.JK', 'PANI.JK', 'PCAR.JK', 'PGUN.JK', 'PMMP.JK',
#                                'PSDN.JK', 'PSGO.JK', 'PTPS.JK', 'RANC.JK', 'ROTI.JK', 'SDPC.JK', 'SGRO.JK', 'SIMP.JK',
#                                'SIPD.JK', 'SKBM.JK', 'SKLT.JK', 'SMAR.JK', 'SOUL.JK', 'SSMS.JK', 'STAA.JK', 'STRK.JK',
#                                'STTP.JK', 'TAPG.JK', 'TAYS.JK', 'TBLA.JK', 'TCID.JK', 'TGKA.JK', 'TGUK.JK', 'TLDN.JK',
#                                'TRGU.JK', 'UCID.JK', 'UDNG.JK', 'ULTJ.JK', 'UNSP.JK', 'UNVR.JK', 'VICI.JK', 'WAPO.JK',
#                                'WICO.JK', 'WIIM.JK', 'WINE.JK', 'WMPP.JK', 'WMUU.JK'],
#     'energy': ['ABMM.JK', 'ADMR.JK', 'ADRO.JK', 'AIMS.JK', 'AKRA.JK', 'ALII.JK', 'APEX.JK', 'ARII.JK', 'ARTI.JK',
#                'BBRM.JK', 'BESS.JK', 'BIPI.JK', 'BOSS.JK', 'BSML.JK', 'BSSR.JK', 'BULL.JK', 'BUMI.JK', 'BYAN.JK',
#                'CANI.JK', 'CBRE.JK', 'CGAS.JK', 'CNKO.JK', 'COAL.JK', 'CUAN.JK', 'DEWA.JK', 'DOID.JK', 'DSSA.JK',
#                'DWGL.JK', 'ELSA.JK', 'ENRG.JK', 'FIRE.JK', 'GEMS.JK', 'GTBO.JK', 'GTSI.JK', 'HILL.JK', 'HITS.JK',
#                'HRUM.JK', 'HUMI.JK', 'IATA.JK', 'INDY.JK', 'INPS.JK', 'ITMA.JK', 'ITMG.JK', 'JSKY.JK', 'KKGI.JK',
#                'KOPI.JK', 'LEAD.JK', 'MAHA.JK', 'MBAP.JK', 'MBSS.JK', 'MCOL.JK', 'MEDC.JK', 'MKAP.JK', 'MTFN.JK',
#                'MYOH.JK', 'PGAS.JK', 'PKPK.JK', 'PSSI.JK', 'PTBA.JK', 'PTIS.JK', 'PTRO.JK', 'RAJA.JK', 'RGAS.JK',
#                'RIGS.JK', 'RMKE.JK', 'RMKO.JK', 'RUIS.JK', 'SEMA.JK', 'SGER.JK', 'SHIP.JK', 'SICO.JK', 'SMMT.JK',
#                'SMRU.JK', 'SOCI.JK', 'SUGI.JK', 'SUNI.JK', 'SURE.JK', 'TAMU.JK', 'TCPI.JK', 'TEBE.JK', 'TOBA.JK',
#                'TPMA.JK', 'TRAM.JK', 'UNIQ.JK', 'WINS.JK', 'WOWS.JK'],
#     'financials': ['ABDA.JK', 'ADMF.JK', 'AGRO.JK', 'AGRS.JK', 'AHAP.JK', 'AMAG.JK', 'AMAR.JK', 'AMOR.JK', 'APIC.JK',
#                    'ARTO.JK', 'ASBI.JK', 'ASDM.JK', 'ASJT.JK', 'ASMI.JK', 'ASRM.JK', 'BABP.JK', 'BACA.JK', 'BANK.JK',
#                    'BBCA.JK', 'BBHI.JK', 'BBKP.JK', 'BBLD.JK', 'BBMD.JK', 'BBNI.JK', 'BBRI.JK', 'BBSI.JK', 'BBTN.JK',
#                    'BBYB.JK', 'BCAP.JK', 'BCIC.JK', 'BDMN.JK', 'BEKS.JK', 'BFIN.JK', 'BGTG.JK', 'BHAT.JK', 'BINA.JK',
#                    'BJBR.JK', 'BJTM.JK', 'BKSW.JK', 'BMAS.JK', 'BMRI.JK', 'BNBA.JK', 'BNGA.JK', 'BNII.JK', 'BNLI.JK',
#                    'BPFI.JK', 'BPII.JK', 'BRIS.JK', 'BSIM.JK', 'BSWD.JK', 'BTPN.JK', 'BTPS.JK', 'BVIC.JK', 'CASA.JK',
#                    'CFIN.JK', 'DEFI.JK', 'DNAR.JK', 'DNET.JK', 'FUJI.JK', 'GSMF.JK', 'HDFA.JK', 'INPC.JK', 'JMAS.JK',
#                    'LIFE.JK', 'LPGI.JK', 'LPPS.JK', 'MASB.JK', 'MAYA.JK', 'MCOR.JK', 'MEGA.JK', 'MFIN.JK', 'MGNA.JK',
#                    'MREI.JK', 'MTWI.JK', 'NICK.JK', 'NISP.JK', 'NOBU.JK', 'OCAP.JK', 'PADI.JK', 'PALM.JK', 'PANS.JK',
#                    'PEGE.JK', 'PLAS.JK', 'PNBN.JK', 'PNBS.JK', 'PNIN.JK', 'PNLF.JK', 'POLA.JK', 'POOL.JK', 'RELI.JK',
#                    'SDRA.JK', 'SFAN.JK', 'SMMA.JK', 'SRTG.JK', 'STAR.JK', 'TIFA.JK', 'TRIM.JK', 'TRUS.JK', 'TUGU.JK',
#                    'VICO.JK', 'VINS.JK', 'VRNA.JK', 'VTNY.JK', 'WOMF.JK', 'YULE.JK'],
#     'healthcare': ['BMHS.JK', 'CARE.JK', 'DGNS.JK', 'DVLA.JK', 'HALO.JK', 'HEAL.JK', 'IKPM.JK', 'INAF.JK', 'IRRA.JK',
#                    'KAEF.JK', 'KLBF.JK', 'MEDS.JK', 'MERK.JK', 'MIKA.JK', 'MMIX.JK', 'MTMH.JK', 'OMED.JK', 'PEHA.JK',
#                    'PEVE.JK', 'PRAY.JK', 'PRDA.JK', 'PRIM.JK', 'PYFA.JK', 'RSCH.JK', 'RSGK.JK', 'SAME.JK', 'SCPI.JK',
#                    'SIDO.JK', 'SILO.JK', 'SOHO.JK', 'SRAJ.JK', 'SURI.JK', 'TSPC.JK'],
#     'industrials': ['AMFG.JK', 'AMIN.JK', 'APII.JK', 'ARKA.JK', 'ARNA.JK', 'ASGR.JK', 'ASII.JK', 'BHIT.JK', 'BINO.JK',
#                     'BLUE.JK', 'BNBR.JK', 'CAKK.JK', 'CCSI.JK', 'CRSN.JK', 'CTTH.JK', 'DYAN.JK', 'FOLK.JK', 'GPSO.JK',
#                     'HEXA.JK', 'HOPE.JK', 'HYGN.JK', 'IBFN.JK', 'ICON.JK', 'IKAI.JK', 'IKBI.JK', 'IMPC.JK', 'INDX.JK',
#                     'INTA.JK', 'JECC.JK', 'JTPE.JK', 'KBLI.JK', 'KBLM.JK', 'KIAS.JK', 'KING.JK', 'KOBX.JK', 'KOIN.JK',
#                     'KONI.JK', 'KPAL.JK', 'KRAH.JK', 'KUAS.JK', 'LABA.JK', 'LION.JK', 'MARK.JK', 'MDRN.JK', 'MFMI.JK',
#                     'MLIA.JK', 'MUTU.JK', 'NTBK.JK', 'PADA.JK', 'PIPA.JK', 'PTMP.JK', 'SCCO.JK', 'SINI.JK', 'SKRN.JK',
#                     'SMIL.JK', 'SOSS.JK', 'SPTO.JK', 'TIRA.JK', 'TOTO.JK', 'TRIL.JK', 'UNTR.JK', 'VISI.JK', 'VOKS.JK',
#                     'WIDI.JK', 'ZBRA.JK'],
#     'infrastructures': ['ACST.JK', 'ADHI.JK', 'ARKO.JK', 'ASLI.JK', 'BALI.JK', 'BDKR.JK', 'BREN.JK', 'BTEL.JK',
#                         'BUKK.JK', 'CASS.JK', 'CENT.JK', 'CMNP.JK', 'DGIK.JK', 'EXCL.JK', 'FIMP.JK', 'FREN.JK',
#                         'GHON.JK', 'GMFI.JK', 'GOLD.JK', 'HADE.JK', 'IBST.JK', 'IDPR.JK', 'INET.JK', 'IPCC.JK',
#                         'IPCM.JK', 'ISAT.JK', 'JAST.JK', 'JKON.JK', 'JSMR.JK', 'KARW.JK', 'KBLV.JK', 'KEEN.JK',
#                         'KETR.JK', 'KOKA.JK', 'KRYA.JK', 'LAPD.JK', 'LCKM.JK', 'LINK.JK', 'MANG.JK', 'META.JK',
#                         'MORA.JK', 'MPOW.JK', 'MTEL.JK', 'MTPS.JK', 'MTRA.JK', 'NRCA.JK', 'OASA.JK', 'PBSA.JK',
#                         'PGEO.JK', 'PORT.JK', 'POWR.JK', 'PPRE.JK', 'PTDU.JK', 'PTPP.JK', 'PTPW.JK', 'RONY.JK',
#                         'SMKM.JK', 'SSIA.JK', 'SUPR.JK', 'TAMA.JK', 'TBIG.JK', 'TGRA.JK', 'TLKM.JK', 'TOPS.JK',
#                         'TOTL.JK', 'TOWR.JK', 'WEGE.JK', 'WIKA.JK', 'WSKT.JK'],
#     'propertiesandrealestate': ['ADCP.JK', 'AMAN.JK', 'APLN.JK', 'ARMY.JK', 'ASPI.JK', 'ASRI.JK', 'ATAP.JK', 'BAPA.JK',
#                                 'BAPI.JK', 'BBSS.JK', 'BCIP.JK', 'BEST.JK', 'BIKA.JK', 'BIPP.JK', 'BKDP.JK', 'BKSL.JK',
#                                 'BSBK.JK', 'BSDE.JK', 'CBPE.JK', 'CITY.JK', 'COWL.JK', 'CPRI.JK', 'CSIS.JK', 'CTRA.JK',
#                                 'DADA.JK', 'DART.JK', 'DILD.JK', 'DMAS.JK', 'DUTI.JK', 'ELTY.JK', 'EMDE.JK', 'FMII.JK',
#                                 'FORZ.JK', 'GAMA.JK', 'GMTD.JK', 'GPRA.JK', 'GRIA.JK', 'HBAT.JK', 'HOMI.JK', 'INDO.JK',
#                                 'INPP.JK', 'IPAC.JK', 'JRPT.JK', 'KBAG.JK', 'KIJA.JK', 'KOCI.JK', 'LAND.JK', 'LCGP.JK',
#                                 'LPCK.JK', 'LPKR.JK', 'LPLI.JK', 'MDLN.JK', 'MKPI.JK', 'MMLP.JK', 'MPRO.JK', 'MSIE.JK',
#                                 'MTLA.JK', 'MTSM.JK', 'MYRX.JK', 'NASA.JK', 'NIRO.JK', 'NZIA.JK', 'OMRE.JK', 'PAMG.JK',
#                                 'PLIN.JK', 'POLI.JK', 'POLL.JK', 'POSA.JK', 'PPRO.JK', 'PUDP.JK', 'PURI.JK', 'PWON.JK',
#                                 'RBMS.JK', 'RDTX.JK', 'REAL.JK', 'RELF.JK', 'RIMO.JK', 'RISE.JK', 'ROCK.JK', 'RODA.JK',
#                                 'SAGE.JK', 'SATU.JK', 'SMDM.JK', 'SMRA.JK', 'SWID.JK', 'TARA.JK', 'TRIN.JK', 'TRUE.JK',
#                                 'UANG.JK', 'URBN.JK', 'VAST.JK', 'WINR.JK'],
#     'tech': ['AREA.JK', 'ATIC.JK', 'AWAN.JK', 'AXIO.JK', 'BELI.JK', 'BUKA.JK', 'CASH.JK', 'CHIP.JK', 'CYBR.JK',
#              'DCII.JK', 'DIVA.JK', 'DMMX.JK', 'EDGE.JK', 'ELIT.JK', 'EMTK.JK', 'ENVY.JK', 'GLVA.JK', 'GOTO.JK',
#              'HDIT.JK', 'IOTF.JK', 'IRSX.JK', 'JATI.JK', 'KIOS.JK', 'KREN.JK', 'LMAS.JK', 'LUCK.JK', 'MCAS.JK',
#              'MENN.JK', 'MLPT.JK', 'MPIX.JK', 'MSTI.JK', 'MTDL.JK', 'NFCX.JK', 'NINE.JK', 'PGJO.JK', 'PTSN.JK',
#              'RUNS.JK', 'SKYB.JK', 'TECH.JK', 'TFAS.JK', 'TOSK.JK', 'TRON.JK', 'UVCR.JK', 'WGSH.JK', 'WIFI.JK',
#              'WIRG.JK', 'ZYRX.JK'],
#     'transportationandlog': ['AKSI.JK', 'ASSA.JK', 'BIRD.JK', 'BLTA.JK', 'BPTR.JK', 'CMPP.JK', 'DEAL.JK', 'ELPI.JK',
#                              'GIAA.JK', 'GTRA.JK', 'HAIS.JK', 'HATM.JK', 'HELI.JK', 'IMJS.JK', 'JAYA.JK', 'KJEN.JK',
#                              'KLAS.JK', 'LAJU.JK', 'LOPI.JK', 'LRNA.JK', 'MIRA.JK', 'MITI.JK', 'MPXL.JK', 'NELY.JK',
#                              'PPGL.JK', 'PURA.JK', 'RCCC.JK', 'SAFE.JK', 'SAPX.JK', 'SDMU.JK', 'SMDR.JK', 'TAXI.JK',
#                              'TMAS.JK', 'TNCA.JK', 'TRJA.JK', 'TRUK.JK', 'WEHA.JK']}
# keys = ['basic_materials', 'consumer_cyclicals', 'consumer_non_cyclicals', 'energy', 'financials', 'healthcare',
#         'industrials', 'infrastructures', 'propertiesandrealestate', 'tech', 'transportationandlog']
#
# df = pd.read_pickle('./storage/all_data(next_day_change)')
#
# dfs_by_year = {}
# for year, group in df.groupby(df.index.year):
#     dfs_by_year[year] = group['Next Day Change']['Adj Close']
#
# # for stock in industries[keys[4]]:
# #     for year, df in dfs_by_year.items():
# #         filter1 = df[stock].dropna()
# #         filter2 = filter1[(filter1 >= -5) & (filter1<= 5)]
# #         plt.plot(filter2.index, filter2)
# dict = {}
# for n in range(11):
#     dict[n] = keys[n]
#
# for n in range(11):
#     # dict[n], axis = plt.subplots(5, 5)
#     dict[n], axis = plt.subplots(figsize=(10,6))
#     axis.set_title(f'AutoCorr of {dict[n]}')
#     stocks = industries[keys[n]]
#     data_list = []
#     for year, df in dfs_by_year.items():
#         df_clean = df[stocks].dropna(axis=1, how='all')
#         row_avg_series = df_clean.mean(axis=1)
#         row_avg_series = row_avg_series.dropna()
#         row_avg_series = row_avg_series[(row_avg_series < 20) & (row_avg_series > -20)]
#         # row_avg_series = row_avg_series.rolling(window=10).mean()
#         row_avg_series = row_avg_series.dropna()
#         data_list.append(row_avg_series)
#     complete_series = pd.concat(data_list)
#     plot_acf(complete_series, lags=365, ax=axis)
#         # if int(year) > 2000 and int(year) < 2024:
#         #     df_clean = df[stocks].dropna(axis=1, how='all')
#         #     row_avg_series = df_clean.mean(axis=1)
#         #     row_avg_series = row_avg_series.dropna()
#         #     row_avg_series = row_avg_series[(row_avg_series < 20)& (row_avg_series>-20)]
#         #     row_avg_series = row_avg_series.rolling(window=10).mean()
#         #     axis[int(int(year) % 5), math.floor(int(year) / 5 - 400)].plot(row_avg_series.index, row_avg_series)
#         #     axis[int(int(year) % 5), math.floor(int(year) / 5 - 400)].set_title(str(year))
#         #     axis[int(int(year) % 5), math.floor(int(year) / 5 - 400)].set_xticks([])
#
# plt.show()
