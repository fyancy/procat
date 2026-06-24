
from pathlib import Path

from datasets.paths_config import sq_legacy_dir, sq_raw_home


def get_SQ_dir(nc, severity='1', speed='39'):
    """

    :param nc: number of classes
    :param severity: '1', '2', '3' # 1, 2, 3, more and more serious
    :param speed: '09', '1', '29', '39'
    :return: list
    """
    _dir = sq_legacy_dir()
    speed = str(speed)
    severity = str(severity)

    this_dir = []
    if nc == 3:
        print(f'SQ {nc}way-{severity}-{speed}Hz')
        this_dir = [Path(_dir) / r'NC' / f'NC_{speed}.npy',
                    Path(_dir) / r'IF' / f'IF{severity}_{speed}.npy',
                    Path(_dir) / r'OF' / f'OF{severity}_{speed}.npy']
    elif nc == 7:
        print(f'SQ {nc}way-{speed}Hz')
        this_dir = [Path(_dir) / r'NC' / f'NC_{speed}.npy',
                    Path(_dir) / r'IF' / f'IF1_{speed}.npy',
                    Path(_dir) / r'IF' / f'IF2_{speed}.npy',
                    Path(_dir) / r'IF' / f'IF3_{speed}.npy',
                    Path(_dir) / r'OF' / f'OF1_{speed}.npy',
                    Path(_dir) / r'OF' / f'OF2_{speed}.npy',
                    Path(_dir) / r'OF' / f'OF3_{speed}.npy',
                    ]
    return this_dir


home = str(sq_raw_home())
inner1 = {'09': [home + r'\inner1\09\REC3585_ch2.txt', home + r'\inner1\09\REC3586_ch2.txt',
                 home + r'\inner1\09\REC3587_ch2.txt'],
          '19': [home + r'\inner1\19\REC3588_ch2.txt', home + r'\inner1\19\REC3589_ch2.txt',
                 home + r'\inner1\19\REC3590_ch2.txt'],
          '29': [home + r'\inner1\29\REC3591_ch2.txt', home + r'\inner1\29\REC3592_ch2.txt',
                 home + r'\inner1\29\REC3593_ch2.txt'],
          '39': [home + r'\inner1\39\REC3594_ch2.txt', home + r'\inner1\39\REC3595_ch2.txt',
                 home + r'\inner1\39\REC3596_ch2.txt']}

inner2 = {'09': [home + r'\inner2\09\REC3607_ch2.txt', home + r'\inner2\09\REC3608_ch2.txt',
                 home + r'\inner2\09\REC3609_ch2.txt'],
          '19': [home + r'\inner2\19\REC3610_ch2.txt', home + r'\inner2\19\REC3611_ch2.txt',
                 home + r'\inner2\19\REC3612_ch2.txt'],
          '29': [home + r'\inner2\29\REC3613_ch2.txt', home + r'\inner2\29\REC3614_ch2.txt',
                 home + r'\inner2\29\REC3615_ch2.txt'],
          '39': [home + r'\inner2\39\REC3616_ch2.txt', home + r'\inner2\39\REC3617_ch2.txt',
                 home + r'\inner2\39\REC3618_ch2.txt']}

inner3 = {'09': [home + r'\inner3\09\REC3520_ch2.txt', home + r'\inner3\09\REC3521_ch2.txt',
                 home + r'\inner3\09\REC3522_ch2.txt'],
          '19': [home + r'\inner3\19\REC3523_ch2.txt', home + r'\inner3\19\REC3524_ch2.txt',
                 home + r'\inner3\19\REC3525_ch2.txt'],
          '29': [home + r'\inner3\29\REC3526_ch2.txt', home + r'\inner3\29\REC3527_ch2.txt',
                 home + r'\inner3\29\REC3528_ch2.txt'],
          '39': [home + r'\inner3\39\REC3529_ch2.txt', home + r'\inner3\39\REC3530_ch2.txt',
                 home + r'\inner3\39\REC3531_ch2.txt']}

outer1 = {'09': [home + r'\outer1\09\REC3500_ch2.txt', home + r'\outer1\09\REC3501_ch2.txt',
                 home + r'\outer1\09\REC3502_ch2.txt'],
          '19': [home + r'\outer1\19\REC3503_ch2.txt', home + r'\outer1\19\REC3504_ch2.txt',
                 home + r'\outer1\19\REC3505_ch2.txt'],
          '29': [home + r'\outer1\29\REC3506_ch2.txt', home + r'\outer1\29\REC3507_ch2.txt',
                 home + r'\outer1\29\REC3508_ch2.txt'],
          '39': [home + r'\outer1\39\REC3510_ch2.txt', home + r'\outer1\39\REC3511_ch2.txt',
                 home + r'\outer1\39\REC3512_ch2.txt']}

outer2 = {'09': [home + r'\outer2\09\REC3482_ch2.txt', home + r'\outer2\09\REC3483_ch2.txt',
                 home + r'\outer2\09\REC3484_ch2.txt'],
          '19': [home + r'\outer2\19\REC3485_ch2.txt', home + r'\outer2\19\REC3486_ch2.txt',
                 home + r'\outer2\19\REC3487_ch2.txt'],
          '29': [home + r'\outer2\29\REC3488_ch2.txt', home + r'\outer2\29\REC3489_ch2.txt',
                 home + r'\outer2\29\REC3490_ch2.txt'],
          '39': [home + r'\outer2\39\REC3491_ch2.txt', home + r'\outer2\39\REC3492_ch2.txt',
                 home + r'\outer2\39\REC3493_ch2.txt']}

outer3 = {'09': [home + r'\outer3\09\REC3464_ch2.txt', home + r'\outer3\09\REC3465_ch2.txt',
                 home + r'\outer3\09\REC3466_ch2.txt'],
          '19': [home + r'\outer3\19\REC3467_ch2.txt', home + r'\outer3\19\REC3468_ch2.txt',
                 home + r'\outer3\19\REC3469_ch2.txt'],
          '29': [home + r'\outer3\29\REC3470_ch2.txt', home + r'\outer3\29\REC3471_ch2.txt',
                 home + r'\outer3\29\REC3472_ch2.txt'],
          '39': [home + r'\outer3\39\REC3473_ch2.txt', home + r'\outer3\39\REC3474_ch2.txt',
                 home + r'\outer3\39\REC3475_ch2.txt']}

normal = {'09': [home + r'\normal\09\REC3629_ch2.txt', home + r'\normal\09\REC3630_ch2.txt',
                 home + r'\normal\09\REC3631_ch2.txt'],
          '19': [home + r'\normal\19\REC3632_ch2.txt', home + r'\normal\19\REC3633_ch2.txt',
                 home + r'\normal\19\REC3634_ch2.txt'],
          '29': [home + r'\normal\29\REC3635_ch2.txt', home + r'\normal\29\REC3636_ch2.txt',
                 home + r'\normal\29\REC3637_ch2.txt'],
          '39': [home + r'\normal\39\REC3638_ch2.txt', home + r'\normal\39\REC3639_ch2.txt',
                 home + r'\normal\39\REC3640_ch2.txt']}
