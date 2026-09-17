use mars_codec::ifs::{self, Leaf};
use mars_codec::mars_format::Header;
use mars_codec::progressive::{self, ProgressiveError};
use mars_codec::quant::ResidualQstep;
use mars_core::Plane;

fn fixture() -> (Vec<u8>, Header, Vec<Leaf>) {
    let hdr = Header {
        geometry: ifs::Header {
            bits_alfa: 4,
            bits_beta: 7,
            min_size: 4,
            max_size: 4,
            shift: 4,
            width: 8,
            height: 8,
            int_max_alfa: 32,
        },
        residual_qstep: ResidualQstep::LEGACY,
    };
    let leaves = [(0, 0), (4, 0), (0, 4), (4, 4)]
        .into_iter()
        .enumerate()
        .map(|(i, (row, col))| {
            let mut residual = vec![0; 16];
            residual[0] = 4;
            residual[1] = -2;
            Leaf {
                row,
                col,
                size: 4,
                mode: 3,
                qalfa: 8,
                qbeta: 45 + i as u32 * 8,
                isometry: 0,
                dom_row: 0,
                dom_col: 0,
                qgx: 0,
                qgy: 0,
                residual,
            }
        })
        .collect::<Vec<_>>();
    let source = Plane::from_vec(8, 8, (0..64).map(|i| (i * 37 % 256) as u8).collect());
    (
        progressive::encode(&source, &hdr, &leaves).unwrap(),
        hdr,
        leaves,
    )
}

#[test]
fn full_layers_match_ifs_at_each_iteration_count_and_default_is_unchanged() {
    let (bytes, hdr, leaves) = fixture();
    for iterations in [1, 2, 10] {
        let decoded = progressive::decode_with_iterations(&bytes, iterations).unwrap();
        assert_eq!(decoded.layers, 4);
        assert_eq!(
            decoded.image,
            ifs::decode_iterative(&hdr, &leaves, iterations)
        );
        assert!(decoded.image.as_slice().windows(2).any(|w| w[0] != w[1]));
    }
    let one = progressive::decode_with_iterations(&bytes, 1)
        .unwrap()
        .image;
    let ten = progressive::decode_with_iterations(&bytes, 10)
        .unwrap()
        .image;
    assert_ne!(one, ten);
    assert_eq!(progressive::decode(&bytes).unwrap().image, ten);
}

#[test]
fn iteration_count_reaches_every_layer_and_partial_prefix() {
    let (bytes, hdr, mut leaves) = fixture();
    let offsets = progressive::layer_end_offsets(&bytes).unwrap();
    for (i, end) in offsets.into_iter().enumerate() {
        let prefix = &bytes[..end];
        let one = progressive::decode_with_iterations(prefix, 1).unwrap();
        let ten = progressive::decode_with_iterations(prefix, 10).unwrap();
        assert_eq!(one.layers as usize, i + 1);
        assert_eq!(ten.layers, one.layers);
        assert_eq!(progressive::decode(prefix).unwrap().image, ten.image);
        if i < 2 {
            assert_eq!(
                one.image, ten.image,
                "flat approximations settle in one step"
            );
        } else {
            assert_ne!(one.image, ten.image);
        }
    }
    for leaf in &mut leaves {
        leaf.mode = 2;
        leaf.residual.clear();
    }
    for iterations in [1, 10] {
        let expected = ifs::decode_iterative(&hdr, &leaves, iterations);
        for end in [offsets[2], offsets[3] - 1] {
            let decoded = progressive::decode_with_iterations(&bytes[..end], iterations).unwrap();
            assert_eq!(decoded.layers, 3);
            assert_eq!(decoded.image, expected);
        }
    }
}

#[test]
fn zero_iterations_are_rejected() {
    let (bytes, _, _) = fixture();
    assert_eq!(
        progressive::decode_with_iterations(&bytes, 0).unwrap_err(),
        ProgressiveError::InvalidIterations
    );
}
