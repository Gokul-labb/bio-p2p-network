use serde::{Deserialize, Serialize};
use std::fmt;
use crate::{ProtocolError, Result};

/// Hierarchical network address inspired by territorial animal navigation
/// 
/// Format: XXX.XXX.Y where:
/// - First XXX: Region identifier (0-999) 
/// - Second XXX: Address group within region (0-999)
/// - Y: Node within group (0-9)
/// 
/// This provides 10,000 regions × 1,000 groups × 10 nodes = 100 billion addressable nodes
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NetworkAddress {
    /// Region identifier (0-999)
    pub region: u16,
    /// Address group within region (0-999) 
    pub group: u16,
    /// Node within group (0-9)
    pub node: u8,
}

impl NetworkAddress {
    /// Create a new network address with validation
    pub fn new(region: u16, group: u16, node: u8) -> Result<Self> {
        if region > 999 {
            return Err(ProtocolError::InvalidAddress {
                address: format!("region {} exceeds maximum 999", region),
            });
        }
        
        if group > 999 {
            return Err(ProtocolError::InvalidAddress {
                address: format!("group {} exceeds maximum 999", group),
            });
        }
        
        if node > 9 {
            return Err(ProtocolError::InvalidAddress {
                address: format!("node {} exceeds maximum 9", node),
            });
        }
        
        Ok(Self { region, group, node })
    }
    
    /// Parse address from string format "XXX.XXX.Y"
    pub fn from_string(address: &str) -> Result<Self> {
        let parts: Vec<&str> = address.split('.').collect();
        
        if parts.len() != 3 {
            return Err(ProtocolError::InvalidAddress {
                address: format!("expected format XXX.XXX.Y, got {}", address),
            });
        }
        
        let region = parts[0].parse::<u16>().map_err(|_| {
            ProtocolError::InvalidAddress {
                address: format!("invalid region in {}", address),
            }
        })?;
        
        let group = parts[1].parse::<u16>().map_err(|_| {
            ProtocolError::InvalidAddress {
                address: format!("invalid group in {}", address),
            }
        })?;
        
        let node = parts[2].parse::<u8>().map_err(|_| {
            ProtocolError::InvalidAddress {
                address: format!("invalid node in {}", address),
            }
        })?;
        
        Self::new(region, group, node)
    }
    
    /// Get the address group identifier (region.group)
    pub fn group_address(&self) -> GroupAddress {
        GroupAddress {
            region: self.region,
            group: self.group,
        }
    }
    
    /// Check if this address is in the same group as another
    pub fn same_group(&self, other: &NetworkAddress) -> bool {
        self.region == other.region && self.group == other.group
    }
    
    /// Check if this address is in the same region as another
    pub fn same_region(&self, other: &NetworkAddress) -> bool {
        self.region == other.region
    }
    
    /// Calculate distance to another address (for proximity-based routing)
    pub fn distance_to(&self, other: &NetworkAddress) -> u32 {
        if self == other {
            return 0;
        }
        
        if self.same_group(other) {
            // Same group, different node
            (self.node as i16 - other.node as i16).abs() as u32
        } else if self.same_region(other) {
            // Same region, different group
            100 + (self.group as i32 - other.group as i32).abs() as u32
        } else {
            // Different region
            10000 + (self.region as i32 - other.region as i32).abs() as u32
        }
    }
    
    /// Get all addresses in the same group
    pub fn group_peers(&self) -> Vec<NetworkAddress> {
        (0..=9)
            .filter_map(|node| {
                if node == self.node {
                    None
                } else {
                    Some(NetworkAddress {
                        region: self.region,
                        group: self.group,
                        node,
                    })
                }
            })
            .collect()
    }
    
    /// Generate a random address (for testing)
    pub fn random() -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        Self {
            region: rng.gen_range(0..=999),
            group: rng.gen_range(0..=999),
            node: rng.gen_range(0..=9),
        }
    }
}

impl fmt::Display for NetworkAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:03}.{:03}.{}", self.region, self.group, self.node)
    }
}

/// Group address for hierarchical routing
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GroupAddress {
    pub region: u16,
    pub group: u16,
}

impl GroupAddress {
    /// Create new group address
    pub fn new(region: u16, group: u16) -> Result<Self> {
        if region > 999 || group > 999 {
            return Err(ProtocolError::InvalidAddress {
                address: format!("invalid group address {}.{}", region, group),
            });
        }
        Ok(Self { region, group })
    }
    
    /// Get all node addresses in this group
    pub fn node_addresses(&self) -> Vec<NetworkAddress> {
        (0..=9)
            .map(|node| NetworkAddress {
                region: self.region,
                group: self.group,
                node,
            })
            .collect()
    }
}

impl fmt::Display for GroupAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:03}.{:03}", self.region, self.group)
    }
}

/// Address range for routing and discovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddressRange {
    pub start: NetworkAddress,
    pub end: NetworkAddress,
}

impl AddressRange {
    /// Check if an address is within this range
    pub fn contains(&self, address: &NetworkAddress) -> bool {
        address >= &self.start && address <= &self.end
    }
    
    /// Create range for entire region
    pub fn region(region: u16) -> Result<Self> {
        Ok(Self {
            start: NetworkAddress::new(region, 0, 0)?,
            end: NetworkAddress::new(region, 999, 9)?,
        })
    }
    
    /// Create range for entire group
    pub fn group(region: u16, group: u16) -> Result<Self> {
        Ok(Self {
            start: NetworkAddress::new(region, group, 0)?,
            end: NetworkAddress::new(region, group, 9)?,
        })
    }
}

/// Implement ordering for addresses (for routing algorithms)
impl Ord for NetworkAddress {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.region, self.group, self.node).cmp(&(other.region, other.group, other.node))
    }
}

impl PartialOrd for NetworkAddress {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_address_creation() {
        let addr = NetworkAddress::new(123, 456, 7).unwrap();
        assert_eq!(addr.region, 123);
        assert_eq!(addr.group, 456);
        assert_eq!(addr.node, 7);
    }
    
    #[test]
    fn test_address_validation() {
        // Valid addresses
        assert!(NetworkAddress::new(0, 0, 0).is_ok());
        assert!(NetworkAddress::new(999, 999, 9).is_ok());
        
        // Invalid addresses
        assert!(NetworkAddress::new(1000, 0, 0).is_err());
        assert!(NetworkAddress::new(0, 1000, 0).is_err());
        assert!(NetworkAddress::new(0, 0, 10).is_err());
    }
    
    #[test]
    fn test_address_parsing() {
        let addr = NetworkAddress::from_string("123.456.7").unwrap();
        assert_eq!(addr.region, 123);
        assert_eq!(addr.group, 456);
        assert_eq!(addr.node, 7);
        
        // Invalid formats
        assert!(NetworkAddress::from_string("123.456").is_err());
        assert!(NetworkAddress::from_string("123.456.7.8").is_err());
        assert!(NetworkAddress::from_string("abc.def.g").is_err());
    }
    
    #[test]
    fn test_address_display() {
        let addr = NetworkAddress::new(1, 23, 4).unwrap();
        assert_eq!(addr.to_string(), "001.023.4");
        
        let addr2 = NetworkAddress::new(999, 0, 9).unwrap();
        assert_eq!(addr2.to_string(), "999.000.9");
    }
    
    #[test]
    fn test_address_relationships() {
        let addr1 = NetworkAddress::new(1, 2, 3).unwrap();
        let addr2 = NetworkAddress::new(1, 2, 4).unwrap(); // Same group
        let addr3 = NetworkAddress::new(1, 3, 3).unwrap(); // Same region
        let addr4 = NetworkAddress::new(2, 2, 3).unwrap(); // Different region
        
        assert!(addr1.same_group(&addr2));
        assert!(!addr1.same_group(&addr3));
        assert!(!addr1.same_group(&addr4));
        
        assert!(addr1.same_region(&addr2));
        assert!(addr1.same_region(&addr3));
        assert!(!addr1.same_region(&addr4));
    }
    
    #[test]
    fn test_distance_calculation() {
        let addr1 = NetworkAddress::new(1, 2, 3).unwrap();
        let addr2 = NetworkAddress::new(1, 2, 4).unwrap(); // Same group
        let addr3 = NetworkAddress::new(1, 3, 3).unwrap(); // Same region
        let addr4 = NetworkAddress::new(2, 2, 3).unwrap(); // Different region
        
        assert_eq!(addr1.distance_to(&addr1), 0);
        assert_eq!(addr1.distance_to(&addr2), 1); // Same group
        assert_eq!(addr1.distance_to(&addr3), 101); // Same region
        assert_eq!(addr1.distance_to(&addr4), 10001); // Different region
    }
    
    #[test]
    fn test_group_peers() {
        let addr = NetworkAddress::new(1, 2, 3).unwrap();
        let peers = addr.group_peers();
        
        assert_eq!(peers.len(), 9); // 10 total - 1 (self)
        assert!(!peers.iter().any(|p| p.node == 3)); // Self not included
        assert!(peers.iter().all(|p| p.region == 1 && p.group == 2));
    }
    
    #[test]
    fn test_address_range() {
        let range = AddressRange::region(5).unwrap();
        let addr_in = NetworkAddress::new(5, 100, 5).unwrap();
        let addr_out = NetworkAddress::new(6, 100, 5).unwrap();
        
        assert!(range.contains(&addr_in));
        assert!(!range.contains(&addr_out));
    }
}