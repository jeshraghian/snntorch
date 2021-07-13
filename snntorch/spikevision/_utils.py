import struct
import numpy as np
import os

# Adapted from https://github.com/nmi-lab/torchneuromorphic by Emre Neftci and Clemens Schaefer
# Which was adapted from https://github.com/gorchard/event-Python/blob/master/eventvision.py by Garrick Orchard


def load_ATIS_bin(filename):
    """Reads in the TD events contained in the N-MNIST and N-CALTECH101 dataset files specified by 'filename'"""
    f = open(filename, "rb")
    raw_data = np.fromfile(f, dtype=np.uint8)
    f.close()
    raw_data = np.uint32(raw_data)

    all_y = raw_data[1::5]
    all_x = raw_data[0::5]
    all_p = (raw_data[2::5] & 128) >> 7  # bit 7
    all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])

    # Process time stamp overflow events
    time_increment = 2 ** 13
    overflow_indices = np.where(all_y == 240)[0]
    for overflow_index in overflow_indices:
        all_ts[overflow_index:] += time_increment

    # Everything else is a proper td spike
    td_indices = np.where(all_y != 240)[0]
    return all_ts, all_x, all_y, all_p


def load_jaer(
    datafile="/tmp/aerout.dat", length=0, version="aedat", debug=1, camera="DVS128"
):
    """
    Load AER data and parse these properties of AE events:
    * timestamps (in microseconds),
    * x, y-position [0..127],
    * polarity (0/1)

    :param datafile: path to the file to read, defaults to ``"/tmp/aerout.dat"``
    :type datafile: string, optional

    :param length: how many bytes should be read, defaults to 0 (whole file)
    :type length: int, optional

    :param version: which file format version is used ("aedat" for v2 or "dat" for v1), defaults to ``"aedat"``
    :type version: string, optional

    :param debug: 0 = silent, 1 = print summary, >=2 = print all debug, defaults to ``1``
    :type debug: int, optional

    :param camera: which event-based camera is used ("DVS128" or "DAVIS240"), defaults to ``"DVS128"11
    :type camera: string, optional

    :return: (ts, xpos, ypos, pol) 4-typle of lists containing data of all events
    :rtype: tuple

    """
    # constants
    aeLen = 8  # 1 AE event takes 8 bytes
    readMode = ">II"  # struct.unpack(), 2x ulong, 4B+4B
    td = 0.000001  # timestep is 1us
    if camera == "DVS128":
        xmask = 0x00FE
        xshift = 1
        ymask = 0x7F00
        yshift = 8
        pmask = 0x1
        pshift = 0
    elif camera == "DAVIS240":  # values take from scripts/matlab/getDVS*.m
        xmask = 0x003FF000
        xshift = 12
        ymask = 0x7FC00000
        yshift = 22
        pmask = 0x800
        pshift = 11
        eventtypeshift = 31
    else:
        raise ValueError("Unsupported camera: %s" % (camera))

    if version == "dat":
        print("using the old .dat format")
        aeLen = 6
        readMode = ">HI"  # ushot, ulong = 2B+4B

    aerdatafh = open(datafile, "rb")
    k = 0  # line number
    p = 0  # pointer, position on bytes
    statinfo = os.stat(datafile)
    if length == 0:
        length = statinfo.st_size

    # header
    lt = aerdatafh.readline()
    while lt and lt[0] == "#":
        p += len(lt)
        k += 1
        lt = aerdatafh.readline()
        if debug >= 2:
            print(str(lt))
        continue

    # variables to parse
    timestamps = []
    xaddr = []
    yaddr = []
    pol = []

    # read data-part of file
    aerdatafh.seek(p)
    s = aerdatafh.read(aeLen)
    p += aeLen

    # print (xmask, xshift, ymask, yshift, pmask, pshift)
    while p < length:
        addr, ts = struct.unpack(readMode, s)
        # parse event type
        if camera == "DAVIS240":
            eventtype = addr >> eventtypeshift
        else:  # DVS128
            eventtype = 0

        # parse event's data
        if eventtype == 0:  # this is a DVS event
            x_addr = (addr & xmask) >> xshift
            y_addr = (addr & ymask) >> yshift
            a_pol = (addr & pmask) >> pshift

            if debug >= 3:
                print("ts->", ts)  # ok
                print("x-> ", x_addr)
                print("y-> ", y_addr)
                print("pol->", a_pol)

            timestamps.append(ts)
            xaddr.append(x_addr)
            yaddr.append(y_addr)
            pol.append(a_pol)

        aerdatafh.seek(p)
        s = aerdatafh.read(aeLen)
        p += aeLen

    if debug > 0:
        try:
            print(
                "read %i (~ %.2fM) AE events, duration= %.2fs"
                % (
                    len(timestamps),
                    len(timestamps) / float(10 ** 6),
                    (timestamps[-1] - timestamps[0]) * td,
                )
            )
            n = 5
            print("showing first %i:" % (n))
            print(
                "timestamps: %s \nX-addr: %s\nY-addr: %s\npolarity: %s"
                % (timestamps[0:n], xaddr[0:n], yaddr[0:n], pol[0:n])
            )
        except:
            print("failed to print statistics")

    return np.array(timestamps), np.array(xaddr), np.array(yaddr), np.array(pol)


def plot_frames_imshow(
    images,
    labels=None,
    nim=11,
    avg=50,
    interval=1,
    do1h=True,
    transpose=False,
    label_mapping=None,
):
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    from matplotlib.pyplot import Normalize

    colors = ["red", "black", "green"]
    cmap = LinearSegmentedColormap.from_list("name", colors)
    if avg > images.shape[1]:
        avg = images.shape[1]

    rnge = range(0, np.maximum(images.shape[1] // avg, 1), interval)

    import pylab as plt

    plt.figure(figsize=[nim + 2, 16])
    import matplotlib.gridspec as gridspec

    if not transpose:
        gs = gridspec.GridSpec(len(rnge), nim)
    else:
        gs = gridspec.GridSpec(nim, len(rnge))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=0.95, wspace=0.0, hspace=0.04)
    if labels is not None:
        if do1h:
            categories = labels.argmax(axis=1)
        else:
            categories = labels
    else:
        categories = range(len(images))
    s = []
    for j in range(nim):
        norm = Normalize(-0.1, 0.1)
        for e, i in enumerate(rnge):
            if not transpose:
                ax = plt.subplot(gs[e, j])
            else:
                ax = plt.subplot(gs[j, e])
            plt.imshow(
                images[j, i * avg : (i * avg + avg), 0, :, :].mean(axis=0).T
                - images[j, i * avg : (i * avg + avg), 1, :, :].mean(axis=0).T,
                cmap=cmap,
                norm=norm,
            )
            plt.xticks([])

            if i == 0 and label_mapping is not None:
                plt.title(label_mapping[int(categories[j])], fontsize=10)
            plt.yticks([])
        s.append(images[j].sum())


def legacy_aedat_to_events(filename, normalize_time=True):
    """
    Uses the dv package to extract events from aedat 2 and aedat 3
    """
    from dv import LegacyAedatFile

    events = []

    with LegacyAedatFile(filename) as f:
        for event in f:
            events.append([event.timestamp, event.polarity, event.x, event.y])

    events = np.column_stack(np.array(events, dtype="uint32")).T
    if normalize_time:
        events[:, 0] -= events[0, 0]

    return events


def aedat_to_events(filename):
    '''
    Used for aedat 3.1
    '''
    label_filename = filename[:-6] +'_labels.csv'
    labels = np.loadtxt(label_filename, skiprows=1, delimiter=',',dtype='uint32')
    events=[]
    with open(filename, 'rb') as f:
        for i in range(5):
            f.readline()
        while True:
            data_ev_head = f.read(28)
            if len(data_ev_head)==0: break

            eventtype = struct.unpack('H', data_ev_head[0:2])[0]
            eventsource = struct.unpack('H', data_ev_head[2:4])[0]
            eventsize = struct.unpack('I', data_ev_head[4:8])[0]
            eventoffset = struct.unpack('I', data_ev_head[8:12])[0]
            eventtsoverflow = struct.unpack('I', data_ev_head[12:16])[0]
            eventcapacity = struct.unpack('I', data_ev_head[16:20])[0]
            eventnumber = struct.unpack('I', data_ev_head[20:24])[0]
            eventvalid = struct.unpack('I', data_ev_head[24:28])[0]

            if(eventtype == 1):
                event_bytes = np.frombuffer(f.read(eventnumber*eventsize), 'uint32')
                event_bytes = event_bytes.reshape(-1,2)

                x = (event_bytes[:,0] >> 17) & 0x00001FFF
                y = (event_bytes[:,0] >> 2 ) & 0x00001FFF
                p = (event_bytes[:,0] >> 1 ) & 0x00000001
                t = event_bytes[:,1]
                events.append([t,x,y,p])

            else:
                f.read(eventnumber*eventsize)
    events = np.column_stack(events)
    events = events.astype('uint32')
    clipped_events = np.zeros([4,0],'uint32')
    for l in labels:
        start = np.searchsorted(events[0,:], l[1])
        end = np.searchsorted(events[0,:], l[2])
        clipped_events = np.column_stack([clipped_events,events[:,start:end]])
    return clipped_events.T, labels


def rosbag_to_events(filename, topic="/dvs_right/events"):
    try:
        from importRosbag.importRosbag import importRosbag
    except ImportError as exc:
        print(
            "This function requires the importRosbag library from https://github.com/event-driven-robotics"
        )
        raise (exc)
    all_events = []
    data = importRosbag(filename)[topic]
    data["ts"] -= data["ts"][0]  # align at 0
    data["ts"] *= 1000000.0  # second to microsecond
    return data
